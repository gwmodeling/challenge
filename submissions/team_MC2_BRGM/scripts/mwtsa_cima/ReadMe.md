## Install
To install the wmtsa_cima library, use pip
Form shell or powershell (windows):
>> pip install path_to_wheel_file

** Example **
>> pip install .\dist\wmtsa_cima-0-py3-none-any.whl


## Usage :
library can then be imported in python file :

###### python_file.py #######
from wmtsa_cima import modwt

modwt().modwt( X=,wtf='la8', nlevels='conservative', boudary='reflection', RetainVJ=False)
modwt().cir_shift()
