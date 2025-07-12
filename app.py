import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model

# Load and preprocess image
def load_and_process_image(path, max_dim=512):
    img = Image.open(path)
    img = img.resize((max_dim, max_dim))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.resize(img, (max_dim, max_dim))
    img = tf.expand_dims(img, axis=0)
    return vgg19.preprocess_input(img)

# Deprocess to display
def deprocess_image(img):
    x = img.copy()
    x = x.reshape((x.shape[1], x.shape[2], 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    return np.clip(x, 0, 255).astype('uint8')

# Load content and style image
content_path = 'content.jpg'
style_path = 'style.jpg'

content_img = load_and_process_image(content_path)
style_img = load_and_process_image(style_path)

# Extract features
vgg = vgg19.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Layers to use
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 
                'block4_conv1', 'block5_conv1']

def get_model():
    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    return Model([vgg.input], outputs)

def get_feature_representations(model, content, style):
    style_outputs = model(style)
    content_outputs = model(content)
    style_features = [style_layer for style_layer in style_outputs[:len(style_layers)]]
    content_features = [content_layer for content_layer in content_outputs[len(style_layers):]]
    return style_features, content_features

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights

    model_outputs = model(init_image)

    style_output_features = model_outputs[:len(style_layers)]
    content_output_features = model_outputs[len(style_layers):]

    style_score = 0
    content_score = 0

    for target_style, comb_style in zip(gram_style_features, style_output_features):
        gram_comb_style = gram_matrix(comb_style)
        style_score += tf.reduce_mean((gram_comb_style - target_style)**2)

    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += tf.reduce_mean((comb_content - target_content)**2)

    style_score *= style_weight
    content_score *= content_weight

    total_loss = style_score + content_score
    return total_loss

@tf.function()
def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    return tape.gradient(all_loss, cfg['init_image']), all_loss

# Set config
model = get_model()
style_features, content_features = get_feature_representations(model, content_img, content_img)
gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

init_image = tf.Variable(content_img, dtype=tf.float32)
opt = tf.optimizers.Adam(learning_rate=5.0)

loss_weights = (1e-2, 1e4)  # style, content
cfg = {
    'model': model,
    'loss_weights': loss_weights,
    'init_image': init_image,
    'gram_style_features': gram_style_features,
    'content_features': content_features
}

# Optimize
epochs = 10
steps_per_epoch = 100

for n in range(epochs):
    for m in range(steps_per_epoch):
        grads, loss = compute_grads(cfg)
        opt.apply_gradients([(grads, init_image)])
        init_image.assign(tf.clip_by_value(init_image, -128.0, 128.0))
    print(f"Epoch {n+1}, Loss: {loss}")

# Display result
final_img = deprocess_image(init_image.numpy())
Image.fromarray(final_img).save("output.jpg")
plt.imshow(final_img)
plt.axis('off')
plt.show()
