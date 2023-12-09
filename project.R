# Define the directories
train_dir <- "C:/Users/Lenovo/Downloads/archive 2/train"
test_dir <- "C:/Users/Lenovo/Downloads/archive 2/test"

# Define the image dimensions and number of classes
row <- 48
col <- 48
classes <- 7

# Function to count examples in each class
count_exp <- function(path, set_) {
  dirs <- list.files(path, full.names = TRUE)
  dict_ <- numeric(length(dirs))
  for (i in 1:length(dirs)) {
    dict_[i] <- length(list.files(dirs[i]))
  }
  df <- data.frame(set_ = set_, dict_)
  colnames(df) <- c("set", names(dict_))
  return(df)
}

# Count examples in the training and test sets
train_count <- count_exp(train_dir, "train")
test_count <- count_exp(test_dir, "test")

# Print the results
print(train_count)
print(test_count)

library(jpeg)
getwd()
# Define the directory paths
train_dir <- "C:/Users/Lenovo/Downloads/archive/train/"
test_dir <- "C:/Users/Lenovo/Downloads/archive/test/"

# Count the total number of labels (classes)
total_labels <- length(list.files(train_dir))

# Create a multi-plot layout
par(mfrow = c(5, total_labels), mar = rep(0, 4))

# Loop to display images
for (x in 1:5) {
  for (y in 1:total_labels) {
    class_dir <- list.files(train_dir)[y]
    img_files <- list.files(paste0(train_dir, class_dir))
    img_path <- file.path(train_dir, class_dir, img_files[x])
    img <- readJPEG(img_path)  # Use readPNG() for PNG images
    # Plot the image
    plot(0:1, 0:1, type = "n", axes = FALSE, xlab = "", ylab = "")
    rasterImage(img, 0, 0, 1, 1)
  }
}

# Reset the layout
par(mfrow = c(1, 1))

# Create an empty data frame
df <- data.frame()

# Loop through subdirectories in the train directory and count the number of images in each subdirectory
for (i in list.files(train_dir)) {
  directory <- file.path(train_dir, i)
  num_images <- length(list.files(directory))
  df <- rbind(df, data.frame(Class = i, Total = num_images))
}

# Sort the data frame by the 'Total' column in descending order

#df <- df[order(-df$Total), ]


# Load the necessary packages
library(ggplot2)

# Create a bar plot
custom_colors <- c("red", "blue", "green", "purple", "orange", "yellow", "pink")
ggplot(data = df, aes(x = Class, y = Total, fill = Class)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(x = "Label", y = "Count") +
  ggtitle("Total images of each label in train dataset") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_fill_manual(values = custom_colors)

# Define the directory path
happy_dir <- file.path(train_dir, "happy")

# Initialize empty lists for image dimensions
dim1 <- vector("integer")
dim2 <- vector("integer")

# Loop through image files in the "happy" class
happy_images <- list.files(happy_dir, full.names = TRUE)
for (img_filename in happy_images) {
  img <- readJPEG(img_filename)  # Use readPNG() for PNG images
  d1 <- dim(img)[1]
  d2 <- dim(img)[2]
  dim1 <- c(dim1, d1)
  dim2 <- c(dim2, d2)
}

# Create a data frame
df <- data.frame(dim1 = dim1, dim2 = dim2)

# Create a joint plot
ggplot(data = df, aes(x = dim1, y = dim2)) +
  geom_point() +
  theme_minimal() +
  labs(x = "Dimension 1", y = "Dimension 2") +
  ggtitle("Joint Plot of Image Dimensions in 'happy' Class")
# Load the required packages
library(keras)

# Define image data generators for training and testing
train_gen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = 'nearest'
)
img_shape <- c(mean(dim1), mean(dim2), 1)
test_gen <- image_data_generator(rescale = 1/255)

# Define image data generators for training and testing datasets
train_generator <- flow_images_from_directory(
  directory = train_dir,
  generator = train_gen,
  target_size = c(img_shape[1], img_shape[2]),
  color_mode = "grayscale",
  batch_size = 64,
  class_mode = "categorical",
  shuffle = TRUE
)

test_generator <- flow_images_from_directory(
  directory = test_dir,
  generator = test_gen,
  target_size = c(img_shape[1], img_shape[2]),
  color_mode = "grayscale",
  batch_size = 64,
  class_mode = "categorical",
  shuffle = FALSE
)

library(reticulate)
version<-"3.9.6"
install_python(version = version)
virtualenv_create("my-python", python_version= version)
use_virtualenv("my-python", required = TRUE)

py_install("tensorflow")
library(keras)

# Create the model
model <- keras_model_sequential()

model %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same", activation = "relu", input_shape = img_shape) %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.2) %>%
  
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same", activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.2) %>%
  
  layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same", activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.2) %>%
  
  layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same", activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.2) %>%
  
  layer_flatten() %>%
  
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  
  layer_dense(units = 1024, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  
  layer_dense(units = length(list.files(train_dir)), activation = "softmax")

# Display model summary
summary(model)
# Compile the model
#install.packages("keras")
#install.packages("tensorflow")

library(keras)
library(tensorflow)
library(keras)
library(tensorflow)

# Assuming 'model' is already defined
# If not, create your model using appropriate layers and configurations

optimizer <- tf$keras$optimizers$legacy$Adam(learning_rate = 0.0001, decay = 1e-6)

model %>% compile(
  optimizer = optimizer,
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)
steps_per_epoch <- train_generator$n %/% train_generator$batch_size
validation_steps <- test_generator$n %/% test_generator$batch_size
num_epochs <- 20
reticulate::py_install("Pillow")
reticulate::py_install("scipy")

reticulate::py_module_available("PIL")
# Restart R session
quit(save = "no")




history <- model %>%
  fit(
    x = train_generator,
    epochs = num_epochs,
    verbose = 1,
    validation_data = test_generator,
    steps_per_epoch = steps_per_epoch,
    validation_steps = validation_steps
  )
save_model_hdf5(model, "C:\\Users\\Lenovo\\Desktop\\r setup\\model.h5")
# Assuming 'model' is your trained machine learning model

saveRDS(model, file = "C:\\Users\\Lenovo\\Desktop\\r setup\\model.rds")


# Load the model from the HDF5 file
loaded_model <- load_model_hdf5("C:\\Users\\Lenovo\\Desktop\\r setup\\model.h5")
# Evaluate the model
evaluation <- evaluate(
  model,
  test_generator,
  steps = validation_steps  # You might need to set the appropriate number of steps
)
test_acc <- evaluation[[2]]
test_loss <- evaluation[[1]]

cat("Validation accuracy:", sprintf("%.2f%%\n", test_acc * 100))
cat("Validation loss:", test_loss, "\n")

library(ggplot2)

# Assuming 'history' is a variable containing the training history
# Make sure that 'history' has the necessary information, e.g., 'history$metrics'

# Convert the history to a data frame
history_df <- data.frame(
  epoch = seq_along(history$metrics$accuracy),
  acc = history$metrics$accuracy,
  val_acc = history$metrics$val_accuracy,
  loss = history$metrics$loss,
  val_loss = history$metrics$val_loss
)

# Plotting accuracy
ggplot(history_df, aes(x = epoch)) +
  geom_line(aes(y = acc, color = "Training"), size = 1) +
  geom_line(aes(y = val_acc, color = "Validation"), size = 1) +
  labs(title = "Training and Validation Accuracy", x = "Epoch", y = "Accuracy") +
  scale_color_manual(values = c("Training" = "green", "Validation" = "red"))

# Plotting loss
ggplot(history_df, aes(x = epoch)) +
  geom_line(aes(y = loss, color = "Training"), size = 1) +
  geom_line(aes(y = val_loss, color = "Validation"), size = 1) +
  labs(title = "Training and Validation Loss", x = "Epoch", y = "Loss") +
  scale_color_manual(values = c("Training" = "green", "Validation" = "red"))

#install.packages("caret")
#install.packages("pROC")
library(caret)
library(pROC)
library(ggplot2)

# Assuming 'model' is your Keras model
# If not, create or load your model using appropriate layers and configurations

# Assuming 'test_generator' is your test data generator
# If not, replace it with your test data

# Predictions
y_pred <- max.col(predict(model, test_generator), ties.method = "random") - 1

# Confusion Matrix
conf_matrix <- confusionMatrix(factor(y_pred), factor(test_generator$classes))

# Print classification report
cat("Classification Report:\n")
print(conf_matrix)

# Plot confusion matrix
cm_df <- as.data.frame(as.table(cm))
colnames(cm) <- levels(test_generator$classes)
rownames(cm) <- levels(test_generator$classes)

# Plot confusion matrix
ggplot(data = str(cm_df), aes(x = colnames(cm), y = rownames(cm))) +
  geom_bar(stat = "identity", fill = "blue")  +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix", x = "Actual", y = "Predicted")
library(imager)

# Load and display the image
image_path <- "C:/Users/Lenovo/Downloads/archive 2/test/fear/PublicTest_98773801.jpg"
img <- load.image(image_path)
plot(img)
# Assuming 'img' is a matrix representing the image in R
# Assuming 'img_path' is the path to the image file
# Assuming 'img_path' is the path to the image file
library(imager)

# Read the image
img <- load.image(img_path)

# Resize the image
resized_img <- resize(img, 48, 48)

# Convert to grayscale and normalize
gray_img <- grayscale(resized_img) / 255.0

# Reshape the image dimension
reshaped_gray <- array_reshape(gray_img, c(1, 48, 48, 1))
# Output the prediction
predicts <- predict_classes(model, reshaped_gray)
label_index <- which.max(predicts)
label <- EMOTIONS[label_index]

predicts <- keras::predict(model, reshaped_gray)
label_index <- which.max(predicts)
label <- EMOTIONS[label_index]


# Print the prediction rates
for (i in seq_along(EMOTIONS)) {
  predictss <- predicts[i]
  cat(sprintf("%-10s prediction rate is   %.2f%%\n", EMOTIONS[i], predictss * 100))
}

cat("\n\n The system considers this expression to be:", label, "\n")
