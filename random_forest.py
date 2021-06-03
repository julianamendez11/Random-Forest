import gdal
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import ogr
from skimage import exposure
from skimage.segmentation import quickshift
from skimage.segmentation import slic
import geopandas as gpd
import scipy
from osgeo import ogr
from tqdm import tqdm

#Se importa la imagen que se va a clasificar
naip_fn = 'C:/Users/imagen.tif'
 
driverTiff = gdal.GetDriverByName('GTiff')
naip_ds = gdal.Open(naip_fn)
nbands = naip_ds.RasterCount

band_data = []
for i in range(1, nbands+1):
    band = naip_ds.GetRasterBand(i).ReadAsArray()
    band_data.append(band)
band_data = np.dstack(band_data)
#Se normaliza la reflectancia de 0 a 1 
img = exposure.rescale_intensity(band_data)
#Se recorta la imagen en segmentos aleatoreos para poder realizar la clasificacion
#segments = quickshift(img, convert2lab=True)
segments=slic(img,n_segments=1000000, compactness=0.1)

#Se crea el raster en donde estaran los segmentos de la imagen 
segments_fn = 'C:/Users/segmentos.tif'
segments_ds = driverTiff.Create(segments_fn, naip_ds.RasterXSize, naip_ds.RasterYSize,1, gdal.GDT_Float32)
segments_ds.SetGeoTransform(naip_ds.GetGeoTransform())
segments_ds.SetProjection(naip_ds.GetProjectionRef())
segments_ds.GetRasterBand(1).WriteArray(segments)
segments_ds = None

print(nbands)

def segment_features(segment_pixels):
    features = []
    npixels, nbands = segment_pixels.shape
    for b in tqdm(range(nbands)):
        stats = scipy.stats.describe(segment_pixels[:, b])
        band_stats = list(stats.minmax) + list(stats)[2:]
        if npixels == 1:
            band_stats[3] = 0.0
        features += band_stats
    return features

segment_ids = np.unique(segments)
objects = []
object_ids = []
for id in tqdm(segment_ids):
    segment_pixels = img[segments == id]
    object_features = segment_features(segment_pixels)
    objects.append(object_features)
    object_ids.append(id)
    
#tqdm libreria para saber cuando se demora un proceso

#Shapes de entrenamiento se leen como un geodataframe
gdf = gpd.read_file('C:/Users/shape entreno.shp')
class_names = gdf['label'].unique()
print('class names', class_names)
class_ids = np.arange(class_names.size) + 1
print('class id', class_ids)

df = pd.DataFrame({'label': class_names, 'id': class_ids})

gdf['id'] = gdf['label'].map(dict(zip(class_names, class_ids)))

#70% de los shapes se asignan al entrenamiento y 30% al testeo.
gdf_train = gdf.sample(frac=0.7) 
gdf_test = gdf.drop(gdf_train.index)
#Se guardan los archivos de entrenamiento y de testeo como shape files
gdf_train.to_file('C:/Users/train_data.shp')
gdf_test.to_file('C:/Users/test_data.shp')

train_fn = 'C:/Users/train_data.shp'
train_ds = ogr.Open(train_fn)
lyr = train_ds.GetLayer()

filas=naip_ds.RasterXSize
columnas=naip_ds.RasterYSize

driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', filas, columnas, 3, gdal.GDT_UInt16)
target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
target_ds.SetProjection(naip_ds.GetProjection())

options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

ground_truth = target_ds.GetRasterBand(1).ReadAsArray()
print('min', ground_truth.min(), 'max', ground_truth.max(), 'mean', ground_truth.mean()) 

ground_truth = target_ds.GetRasterBand(1).ReadAsArray()

classes = np.unique(ground_truth)[1:]

print(classes)

segments_per_class = {}
for klass in classes:
    segments_of_class = segments[ground_truth == klass]
    segments_per_class[klass] = set(segments_of_class)
    print('Segmentos de entrenamiento para la clase', klass, ':', len(segments_of_class))
 

intersection = set()
accum = set()
for class_segments in segments_per_class.values():
    intersection |= accum.intersection(class_segments)
    accum |= class_segments
assert len(intersection) == 0, "Los segmentos representan varias clases y eso implica errores"

train_img = np.copy(segments)
threshold = train_img.max() + 1 


for klass in classes:
    class_label = threshold + klass
    for segment_id in segments_per_class[klass]:
        train_img[train_img == segment_id] = class_label
        
train_img[train_img <= threshold] = 0
train_img[train_img > threshold] -= threshold

training_objects = []
training_labels = []
for klass in classes:
    class_train_object = [v for i, v in enumerate(objects) if segment_ids[i] in segments_per_class[klass]]
    training_labels += [klass] * len(class_train_object)
    training_objects += class_train_object
 

#Random forest clasificador
arboles= RandomForestClassifier(n_jobs=-1) 
arboles.fit(training_objects, training_labels) 
predicted = arboles.predict(objects)  

clf = np.copy(segments)
for segment_id, klass in zip(segment_ids, predicted):
    clf[clf == segment_id] = klass
 
mask = np.sum(img, axis=2)  # para quitar valores malos
mask[mask > 0.0] = 1.0
mask[mask == 0.0] = -1.0
clf = np.multiply(clf, mask)
clf[clf < 0] = -9999.0

#Se guarda la clasificacion a un raster con GDAL
clfds = driverTiff.Create('C:/Users/clasificacion.tif',filas, columnas, 3, gdal.GDT_Float32)
clfds.SetGeoTransform(naip_ds.GetGeoTransform())
clfds.SetProjection(naip_ds.GetProjection())
clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
clfds.GetRasterBand(1).WriteArray(clf)
clfds = None

from sklearn import metrics
from sklearn.metrics import classification_report


test_fn = 'C:/Users/test_data.shp'
test_ds = ogr.Open(test_fn)
lyr = test_ds.GetLayer()
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', naip_ds.RasterXSize, naip_ds.RasterYSize, 3, gdal.GDT_UInt16)
target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
target_ds.SetProjection(naip_ds.GetProjection())
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

truth = target_ds.GetRasterBand(1).ReadAsArray() 
 
pred_ds = gdal.Open('C:/Users/Servi/Documents/class_oil.tif')  
pred = pred_ds.GetRasterBand(1).ReadAsArray()  
idx = np.nonzero(truth)
 

print (classification_report(truth[idx], pred[idx]))
