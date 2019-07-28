# Generate classes

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, help='Path to the class folders')
parser.add_argument('--classfile', required=False, default='', help='Class file to produce as output')
opt = parser.parse_args()

classes = os.listdir(opt.path)
classes = [c for c in classes if os.path.isdir(os.path.join(opt.path,c))]
classes.sort()

print(classes)

if opt.classfile != '':
    with open(opt.classfile, 'w') as f:
        for c in classes:
            f.write("{}\n".format(c))
