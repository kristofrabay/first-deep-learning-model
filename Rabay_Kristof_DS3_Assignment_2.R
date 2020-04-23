## ---- include = F--------------------------------------------------------
library(data.table)
library(keras)
library(ggplot2)

options(digits = 6)
options(scipen = 999)
theme_set(theme_bw())


## ---- warning = F, message = F-------------------------------------------
fashion_mnist <- dataset_fashion_mnist()
x_train <- fashion_mnist$train$x
y_train <- fashion_mnist$train$y
x_test <- fashion_mnist$test$x
y_test <- fashion_mnist$test$y


## ------------------------------------------------------------------------
attributes(x_train)
attributes(x_test)


## ------------------------------------------------------------------------
show_mnist_image <- function(x) {image(1:28, 1:28, t(x)[, nrow(x):1], col = gray((0:255)/255))}


## ---- fig.height = 5, fig.width = 5, fig.align='center'------------------
show_mnist_image(x_train[18, , ])
#x_train[18, , ]


## ---- fig.height = 5, fig.width = 5, fig.align='center'------------------
show_mnist_image(x_train[61, , ])
#x_train[61, , ]


## ------------------------------------------------------------------------
x_train <- array_reshape(x_train, c(dim(x_train)[1], 784)) / 255
x_test <- array_reshape(x_test, c(dim(x_test)[1], 784)) / 255


## ------------------------------------------------------------------------
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


## ---- warning = F, message = F-------------------------------------------
model <- keras_model_sequential() 

model %>% 
  layer_dense(units = 128, activation = 'sigmoid', input_shape = ncol(x_train)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 128, activation = 'sigmoid') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy')



## ------------------------------------------------------------------------
summary(model)



## ---- warning = F, message = F-------------------------------------------

set.seed(20202020)

history <- model %>% fit(x_train, y_train, 
                         epochs = 30, 
                         batch_size = 128, 
                         validation_split = 1/3,
                         verbose = 0,
                         callbacks = list(callback_early_stopping(monitor = 'val_loss', 
                                                                  min_delta = 0.005, 
                                                                  patience = 5)))
                                          #callback_reduce_lr_on_plateau(monitor = 'val_loss', 
                                           #                             patience = 3, 
                                            #                            factor = 0.1)))


## ---- fig.height = 5, fig.width = 8, fig.align='center'------------------
plot(history, method = 'ggplot2', smooth = T, theme_bw = T)


## ------------------------------------------------------------------------
evaluate(model, x_train, y_train, verbose = 0)$accuracy 


## ------------------------------------------------------------------------
evaluate(model, x_test, y_test, verbose = 0)$accuracy 


## ---- fig.height = 8, fig.width = 8, fig.align='center'------------------
predicted_classes_test <- model %>% predict_classes(x_test)
real_classes_test <- as.numeric(fashion_mnist$test$y)

dt_pred_vs_real <- data.table(predicted = predicted_classes_test, real = real_classes_test)

ggplot(dt_pred_vs_real[, .N, by = .(predicted, real)], aes(predicted, real)) +
  geom_tile(aes(fill = N), colour = "white") +
  scale_x_continuous(breaks = 0:9) +
  scale_y_continuous(breaks = 0:9) +
  geom_text(aes(label = sprintf("%1.0f", N)), color = "white") +
  theme_bw() +
  theme(legend.position = "none")


## ---- fig.height = 5, fig.width = 5, fig.align='center'------------------
dt_pred_vs_real[, row_number := 1:.N]

# predicted: 6, real : 0
indices_of_mistakes <- dt_pred_vs_real[predicted == 6 & real == 0,][["row_number"]]

# take one example
ix <- indices_of_mistakes[10]

# show the image
show_mnist_image(fashion_mnist$test$x[ix, , ])


## ------------------------------------------------------------------------

x_train <- fashion_mnist$train$x
y_train <- fashion_mnist$train$y
x_test <- fashion_mnist$test$x
y_test <- fashion_mnist$test$y

x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1)) / 255
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1)) / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)



## ------------------------------------------------------------------------
attributes(x_train)
attributes(x_test)


## ------------------------------------------------------------------------
cnn <- keras_model_sequential()

cnn %>% layer_conv_2d(filters = 32, 
                      kernel_size = c(3, 3), 
                      activation = 'relu', 
                      input_shape = dim(x_train)[2:4]) %>% 
  layer_conv_2d(filters = 32, 
                kernel_size = c(3, 3), 
                activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.2) %>%
  layer_flatten() %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 32, activation = 'relu') %>% 
  layer_dense(units = 10, activation = 'softmax')

cnn %>% compile(loss = 'categorical_crossentropy',
                optimizer = 'adam',
                metrics = 'accuracy')


## ------------------------------------------------------------------------
summary(cnn)


## ---- warning = F, message = F-------------------------------------------
set.seed(20202020)

cnn_hist <- cnn %>% fit(x_train, y_train, 
                         epochs = 20, 
                         batch_size = 128, 
                         validation_split = 1/3,
                         verbose = 0,
                         callbacks = list(callback_early_stopping(monitor = 'val_loss', 
                                                                  min_delta = 0.005, 
                                                                  patience = 5)))
                                          #callback_reduce_lr_on_plateau(monitor = 'val_loss', 
                                           #                             patience = 3, 
                                            #                            factor = 0.1)))


## ---- fig.height = 5, fig.width = 8, fig.align='center'------------------
plot(cnn_hist, method = 'ggplot2', smooth = T, theme_bw = T)


## ------------------------------------------------------------------------
evaluate(cnn, x_train, y_train, verbose = 0)$accuracy 


## ------------------------------------------------------------------------
evaluate(cnn, x_test, y_test, verbose = 0)$accuracy 


## ---- fig.height = 5, fig.width = 8, fig.align='center'------------------
path <- "../../../machine-learning-course/data/hot-dog-not-hot-dog/train/hot_dog/1000288.jpg"
grid::grid.raster(image_to_array(image_load(path, target_size = c(250, 250))) / 255)


## ---- fig.height = 5, fig.width = 8, fig.align='center'------------------
path <- "../../../machine-learning-course/data/hot-dog-not-hot-dog/train/not_hot_dog/100135.jpg"
grid::grid.raster(image_to_array(image_load(path, target_size = c(250, 250))) / 255)


## ---- eval = FALSE, include = TRUE---------------------------------------
## # 249 photos in each
## hotdog_number <- length(list.files("../../../machine-learning-course/data/hot-dog-not-hot-dog_with_validation/train/hot_dog/"))
## nonhotdog_number <- length(list.files("../../../machine-learning-course/data/hot-dog-not-hot-dog_with_validation/train/not_hot_dog/"))
## 
## # 20% of images
## # 49 photos each
## hd_to_val <- as.integer(0.2*hotdog_number)
## nhd_to_val <- as.integer(0.2*nonhotdog_number)
## 
## # select the images randomly
## set.seed(20202020)
## hd <- sample(list.files("../../../machine-learning-course/data/hot-dog-not-hot-dog_with_validation/train/hot_dog/",
##                         full.names = T),
##              hd_to_val,
##              replace = FALSE)
## set.seed(20202020)
## nhd <- sample(list.files("../../../machine-learning-course/data/hot-dog-not-hot-dog_with_validation/train/not_hot_dog/",
##                          full.names = T),
##               nhd_to_val,
##               replace = FALSE)
## 
## # move photos to new destination
## 
## for (hot_dog in hd){
##   file.copy(hot_dog, "../../../machine-learning-course/data/hot-dog-not-hot-dog_with_validation/validation/hot_dog/")
## }
## 
## for (not_hot_dog in nhd){
##   file.copy(not_hot_dog, "../../../machine-learning-course/data/hot-dog-not-hot-dog_with_validation/validation/not_hot_dog/")
## }
## 
## for (hot_dog in hd){
##   file.remove(hot_dog)
## }
## 
## for (not_hot_dog in nhd){
##   file.remove(not_hot_dog)
## }
## 


## ------------------------------------------------------------------------
train_datagen <- image_data_generator(rescale = 1/255) 
valid_datagen <- image_data_generator(rescale = 1/255) 
test_datagen <- image_data_generator(rescale = 1/255) 

image_size <- c(250, 250)
batch_size <- 20

train_generator <- flow_images_from_directory("../../../machine-learning-course/data/hot-dog-not-hot-dog_with_validation/train/",  
                                              train_datagen,
                                              target_size = image_size,
                                              batch_size = batch_size,
                                              class_mode = "binary")

valid_generator <- flow_images_from_directory("../../../machine-learning-course/data/hot-dog-not-hot-dog_with_validation/validation/",  
                                              valid_datagen,
                                              target_size = image_size,
                                              batch_size = batch_size,
                                              class_mode = "binary")

test_generator <- flow_images_from_directory("../../../machine-learning-course/data/hot-dog-not-hot-dog_with_validation/test/", 
                                             test_datagen,
                                             target_size = image_size,
                                             batch_size = batch_size,
                                             class_mode = "binary")


## ------------------------------------------------------------------------
train_generator$samples
valid_generator$samples
test_generator$samples


## ------------------------------------------------------------------------
str(generator_next(train_generator))


## ---- warning = F, message = F-------------------------------------------
noaug <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(250, 250, 3), padding = 'same') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>%
  layer_global_average_pooling_2d() %>% 
  #layer_flatten() %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 1, activation = 'sigmoid')


noaug %>% compile(loss = 'binary_crossentropy',
                optimizer = 'adam', # also tried rmsprop, adam with different lr
                metrics = 'accuracy')


## ------------------------------------------------------------------------
summary(noaug)


## ---- warning = F, message = F-------------------------------------------
set.seed(20202020)
history <- noaug %>% fit_generator(train_generator,
                                    steps_per_epoch = 20,
                                    epochs = 20, 
                                    validation_data = valid_generator, 
                                    validation_steps = 5,
                                    verbose = 0,
                                    callbacks = list(callback_reduce_lr_on_plateau(monitor = 'val_loss',
                                                                                   patience = 3, 
                                                                                   factor = 0.1),
                                                     callback_early_stopping(monitor = 'val_loss', 
                                                                             min_delta = 0.005, 
                                                                             patience = 8)))


## ---- fig.height = 5, fig.width = 8, fig.align='center'------------------
plot(history, method = 'ggplot2', smooth = T, theme_bw = T)


## ------------------------------------------------------------------------
evaluate_generator(noaug, test_generator, 50)$accuracy


## ------------------------------------------------------------------------
train_datagen <- image_data_generator(rescale = 1/255,
                                     rotation_range = 45,
                                     width_shift_range = 0.2,
                                     height_shift_range = 0.2,
                                     zoom_range = 0.2,
                                     horizontal_flip = TRUE,
                                     vertical_flip = TRUE,
                                     fill_mode = "nearest")

train_generator <- flow_images_from_directory("../../../machine-learning-course/data/hot-dog-not-hot-dog_with_validation/train/",  
                                              train_datagen,
                                              target_size = image_size,
                                              batch_size = batch_size,
                                              class_mode = "binary")



## ---- warning = F, message = F-------------------------------------------
aug <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(250, 250, 3), padding = 'same') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>%
  layer_global_average_pooling_2d() %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 1, activation = 'sigmoid')


aug %>% compile(loss = 'binary_crossentropy',
                optimizer = 'adam', # also tried rmsprop, adam with different lr
                metrics = 'accuracy')


## ---- warning = F, message = F-------------------------------------------
set.seed(20202020)
history_aug <- aug %>% fit_generator(train_generator,
                                    steps_per_epoch = 20,
                                    epochs = 20, 
                                    validation_data = valid_generator, 
                                    validation_steps = 5,
                                    verbose = 0,
                                    callbacks = list(callback_reduce_lr_on_plateau(monitor = 'val_loss',
                                                                                   patience = 3, 
                                                                                   factor = 0.1),
                                                     callback_early_stopping(monitor = 'val_loss', 
                                                                             min_delta = 0.005, 
                                                                             patience = 8)))


## ---- fig.height = 5, fig.width = 8, fig.align='center'------------------
plot(history_aug, method = 'ggplot2', smooth = T, theme_bw = T)


## ------------------------------------------------------------------------
evaluate_generator(aug, test_generator, 50)$accuracy


## ---- warning = F, message = F-------------------------------------------
base_model <- application_inception_v3(include_top = FALSE, weights = 'imagenet', input_shape = c(250, 250, 3))

freeze_weights(base_model)

top_layer <- base_model$output %>% 
  layer_global_average_pooling_2d() %>% 
  layer_dense(units = 32, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

tr_model <- keras_model(inputs = base_model$input, outputs = top_layer)

tr_model %>% compile(loss = 'binary_crossentropy',
                optimizer = 'adam', 
                metrics = 'accuracy')



## ------------------------------------------------------------------------
summary(tr_model)


## ---- warning = F, message = F-------------------------------------------
set.seed(20202020)
tlhis <- tr_model %>% fit_generator(train_generator,
                                    steps_per_epoch = 20,
                                    epochs = 5, 
                                    validation_data = valid_generator, 
                                    validation_steps = 5)
                                    #callbacks = list(callback_reduce_lr_on_plateau(monitor = 'val_loss',
                                     #                                              patience = 3, 
                                      #                                             factor = 0.1),
                                                     #callback_early_stopping(monitor = 'val_loss', 
                                                     #                        min_delta = 0.005, 
                                                     #                        patience = 8)))


## ---- fig.height = 5, fig.width = 8, fig.align='center'------------------
plot(tlhis, method = 'ggplot2', smooth = T, theme_bw = T)


## ------------------------------------------------------------------------
evaluate_generator(tr_model, test_generator, 50)$accuracy

