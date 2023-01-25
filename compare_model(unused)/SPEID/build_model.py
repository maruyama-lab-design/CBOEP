# Keras imports
from tensorflow.keras.layers import Input, Convolution1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, LSTM, Activation, Bidirectional
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Concatenate, Average, concatenate

# model parameters
enhancer_length = 3000 # TODO: get this from input
promoter_length = 2000 # TODO: get this from input
n_kernels = 200 # Number of kernels; used to be 1024
filter_length = 40 # Length of each kernel
LSTM_out_dim = 50 # Output direction of ONE DIRECTION of LSTM; used to be 512
dense_layer_size = 800

# Convolutional/maxpooling layers to extract prominent motifs
# Separate identically initialized convolutional layers are trained for
# enhancers and promoters
# Define enhancer layers
enhancer_conv_layer = Convolution1D(    input_shape = (enhancer_length, 4),
										filters =  n_kernels,
										kernel_size = filter_length,
										padding = "valid",
										kernel_regularizer=l2(1e-5)
										)
enhancer_max_pool_layer = MaxPooling1D(pool_size = filter_length//2, strides = filter_length//2)

# Build enhancer branch
enhancer_branch = Sequential()
enhancer_branch.add(enhancer_conv_layer)
enhancer_branch.add(Activation("relu"))
enhancer_branch.add(enhancer_max_pool_layer)

# Define promoter layers branch:

promoter_conv_layer = Convolution1D( 
										input_shape = (promoter_length, 4),
										# nb_filter = n_kernels,
										filters =  n_kernels,
										# filter_length = filter_length,
										kernel_size = filter_length,
										# border_mode = "valid",
										padding = "valid",
										# subsample_length = 1,
										# W_regularizer = l2(1e-5)
										kernel_regularizer=l2(1e-5)
										)
promoter_max_pool_layer = MaxPooling1D(pool_size = filter_length//2, strides = filter_length//2)

# Build promoter branch
promoter_branch = Sequential()
promoter_branch.add(promoter_conv_layer)
promoter_branch.add(Activation("relu"))
promoter_branch.add(promoter_max_pool_layer)

# Define main model layers
# Concatenate outputs of enhancer and promoter convolutional layers
merge_layer = Concatenate([enhancer_branch, promoter_branch])


# Bidirectional LSTM to extract combinations of motifs
biLSTM_layer = Bidirectional(LSTM(LSTM_out_dim,
									# output_dim = LSTM_out_dim,
									return_sequences = True))

# Dense layer to allow nonlinearities
dense_layer = Dense(dense_layer_size,
					kernel_initializer = "glorot_uniform",
					kernel_regularizer = l2(1e-6))

# Logistic regression layer to make final binary prediction
LR_classifier_layer = Dense(1)
  

def build_model(use_JASPAR = True):

	input_enhancer = Input(shape=(enhancer_length, 4))
	conv_enhancer = Convolution1D(input_shape = (promoter_length, 4),
										filters =  n_kernels,
										kernel_size = filter_length,
										padding = "valid",
										kernel_regularizer=l2(1e-5)
										)(input_enhancer)
	conv_enhancer = Activation("relu")(conv_enhancer)
	conv_enhancer = MaxPooling1D(pool_size = filter_length//2, strides = filter_length//2)(conv_enhancer)


	input_promoter = Input(shape=(promoter_length, 4))
	conv_promoter = Convolution1D(input_shape = (promoter_length, 4),
										filters =  n_kernels,
										kernel_size = filter_length,
										padding = "valid",
										kernel_regularizer=l2(1e-5)
										)(input_promoter)
	conv_promoter = Activation("relu")(conv_promoter)
	conv_promoter = MaxPooling1D(pool_size = filter_length//2, strides = filter_length//2)(conv_promoter)



	# model = Sequential()
	# model.add(merge_layer)
	merge_layer = concatenate([conv_enhancer, conv_promoter], axis=1)
	# model.add(BatchNormalization())
	model = BatchNormalization()(merge_layer)
	# model.add(Dropout(0.25))
	model = Dropout(0.25)(model)
	# model.add(biLSTM_layer)
	model = biLSTM_layer(model)
	# model.add(BatchNormalization())
	model = BatchNormalization()(model)
	# model.add(Dropout(0.5))
	model = Dropout(0.5)(model)
	# model.add(Flatten())
	model = Flatten()(model)
	# model.add(dense_layer)
	model = dense_layer(model)
	# model.add(BatchNormalization())
	model = BatchNormalization()(model)
	# model.add(Activation("relu"))
	model = Activation("relu")(model)
	# model.add(Dropout(0.5))
	model = Dropout(0.5)(model)
	# model.add(LR_classifier_layer)
	model = LR_classifier_layer(model)
	# model.add(BatchNormalization())
	model = BatchNormalization()(model)
	# model.add(Activation("sigmoid"))
	model = Activation("sigmoid")(model)

	model = Model(inputs=[input_enhancer, input_promoter], outputs=model)
  
  # Read in and initialize convolutional layers with motifs from JASPAR
#   if use_JASPAR:
#     util.initialize_with_JASPAR(enhancer_conv_layer, promoter_conv_layer)

	return model


# model = build_model()
# model.compile()
# model.summary()