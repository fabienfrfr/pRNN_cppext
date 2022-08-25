# fabienfrfr 20220825

from ctypes import *

example = CDLL('build/exampleApp') 

example.main()
