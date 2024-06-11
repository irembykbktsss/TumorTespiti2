import warnings
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, EfficientNetB3, DenseNet121, VGG16, InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning, module='keras.src.trainers.data_adapters.py_dataset_adapter')

# GPU kullanımı
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Veri setinin yüklenmesi ve hazırlanması
train_dir = 'C:/Users/MONSTER/Desktop/veriseti/Training'
test_dir = 'C:/Users/MONSTER/Desktop/veriseti/Testing'

# Eğitim ve doğrulama veri jeneratörlerinin oluşturulması
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest', validation_split=0.2)

# Test veri jeneratörü (sadece rescale)
test_datagen = ImageDataGenerator(rescale=1./255)

# Veri jeneratörlerinin oluşturulması
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224),
                                                    batch_size=32, class_mode='categorical', subset='training')

val_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224),
                                                  batch_size=32, class_mode='categorical', subset='validation')

test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224),
                                                  batch_size=32, class_mode='categorical', shuffle=False)

# Model listesi
models = [
    ('ResNet50', ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))),
    ('EfficientNetB3', EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))),
    ('VGG16', VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))),
    ('InceptionV3', InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))),
    ('DenseNet121', DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
]

results = {}

# Her model için eğitim ve değerlendirme
for model_name, base_model in models:
    print(f'Eğitiliyor: {model_name}')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_generator, epochs=1, validation_data=val_generator, verbose=2)

    # Test seti üzerinde değerlendirme
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f'Test Accuracy for {model_name}: {test_accuracy}')

    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)

    print(f'Classification Report for {model_name}')
    report = classification_report(test_generator.classes, y_pred, target_names=test_generator.class_indices.keys(), output_dict=True)
    print(classification_report(test_generator.classes, y_pred, target_names=test_generator.class_indices.keys()))

    # Confusion matrix
    cm = confusion_matrix(test_generator.classes, y_pred)
    print(f'Confusion Matrix for {model_name}')
    print(cm)

    # Sensitivity, Specificity ve diğer metriklerin hesaplanması
    sensitivity = {}
    specificity = {}
    for i, label in enumerate(test_generator.class_indices.keys()):
        sensitivity[label] = cm[i, i] / np.sum(cm[i, :])
        specificity[label] = np.sum(cm[:, i]) - cm[i, i] / (np.sum(cm) - np.sum(cm[i, :]))

    results[model_name] = {
        'accuracy': report['accuracy'],
        'macro avg f1-score': report['macro avg']['f1-score'],
        'macro avg precision': report['macro avg']['precision'],
        'macro avg recall': report['macro avg']['recall'],
        'sensitivity': sensitivity,
        'specificity': specificity
    }

    # Eğitim ve doğrulama kayıplarının ve doğruluklarının görselleştirilmesi
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f'Loss for {model_name}')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title(f'Accuracy for {model_name}')

    plt.show()

# Modellerin sonuçlarının karşılaştırılması
print("Modellerin Performans Karşılaştırması")
for model_name, metrics in results.items():
    print(f'\nModel: {model_name}')
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Avg Precision: {metrics['macro avg precision']:.4f}")
    print(f"Macro Avg Recall: {metrics['macro avg recall']:.4f}")
    print(f"Macro Avg F1 Score: {metrics['macro avg f1-score']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']}")
    print(f"Specificity: {metrics['specificity']}")
