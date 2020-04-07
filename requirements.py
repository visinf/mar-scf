import sys
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

assert sys.version_info >= (3, 5), 'Insufficent Python version. Requires Python >= 3.5'

print('Python version requirement satisfied!')

dependencies = [
	'torch==1.0.0',
	'torchvision==0.2.2',
 	'numpy>=1.16.1',
 	'scipy>=1.1.0',
 	'tqdm>=4.19.9',
]

pkg_resources.require(dependencies)
print('All required packages found!')