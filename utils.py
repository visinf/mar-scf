import os
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from flow_modules.misc import ShiftTransform, MnistGlowTransform

def get_dataset( dataset_name, batch_size, data_root=None, train_workers=4, test_workers=2 ):

	assert dataset_name in ['cifar10','mnist','imagenet_32','imagenet_64'], "Invalid Dataset Name"

	if dataset_name == 'cifar10':
		if data_root is None:
			data_root = '../cifar_data'

		image_shape = [32,32,3]

		transform_train = transforms.Compose([ 
			ShiftTransform(3),
			transforms.RandomHorizontalFlip(), 
			transforms.ToTensor(),  
			transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

		transform_test = transforms.Compose([ transforms.ToTensor(),  
			transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

		trainset = dsets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=train_workers)

		testset = dsets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,drop_last=True,num_workers=test_workers)

	elif dataset_name == 'mnist':
		if data_root is None:
			data_root = '../cifar_data'

		image_shape = [32,32,3]

		transform_train = transforms.Compose([ 
			MnistGlowTransform(2),
			transforms.ToTensor(),  
			transforms.Normalize((0.5,), (1.0,))])

		transform_test = transforms.Compose([ 
			MnistGlowTransform(2),
			transforms.ToTensor(),  
			transforms.Normalize((0.5,), (1.0,))])

		trainset = dsets.MNIST(root=data_root, train=True, download=True, transform=transform_train)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=train_workers)

		testset = dsets.MNIST(root=data_root, train=False, download=True, transform=transform_test)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,drop_last=True,num_workers=train_workers)

	elif dataset_name == 'imagenet_32':
		if data_root is None:
			data_root = '../imagenet_data/'

		image_shape = [32,32,3]

		transform_train = transforms.Compose([ 
			transforms.ToTensor(),  
			transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

		transform_test = transforms.Compose([ transforms.ToTensor(),  
			transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

		trainset = dsets.ImageFolder(root=os.path.join(data_root,'train_32x32'), transform=transform_train)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=train_workers)

		testset = dsets.ImageFolder(root=os.path.join(data_root,'valid_32x32'), transform=transform_test)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,drop_last=True,num_workers=test_workers)

	elif dataset_name == 'imagenet_64':
		if data_root is None:
			data_root = '../imagenet_data/'

		image_shape = [64,64,3]

		transform_train = transforms.Compose([ 
			transforms.ToTensor(),  
			transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

		transform_test = transforms.Compose([ transforms.ToTensor(),  
			transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

		trainset = dsets.ImageFolder(root=os.path.join(data_root,'train_64x64'), transform=transform_train)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=train_workers)

		testset = dsets.ImageFolder(root=os.path.join(data_root,'valid_64x64'), transform=transform_test)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,drop_last=True,num_workers=test_workers)

	return train_loader, test_loader, image_shape
