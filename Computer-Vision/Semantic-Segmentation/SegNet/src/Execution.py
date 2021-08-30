from Train import Train
from Test import Test

train = Train()
test = Test()
train.load_model("<Best Model Path>", 'cuda')
training_loader, validation_loader, testing_loader, visual_loader = train.fetch_data_loaders()
train.visualize_original_data()
train.get_model_parameters()
train.train_model()
test.test_model()