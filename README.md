# AI4ALL project

## To do
- Try more pics from outside leafsnap (the models do well on leafsnap but not so well outside it)
- Try out data augmentation and other improvements
- Create a concrete list of student deliverables

## Help
To train a new alexnet model:

python alexnet.py --outpath [path/for/saving/model.pth] --device [cpu | cuda:0 | cuda:1 | ... cuda:N] --loadfrom [/path/to/existing/model.pth] --nepochs [epochs to train for] --batchsize [size] --lr [learning rate] > path/to/log.txt

To test on new images:
Download or take images, then run

python test_on_new.py --loadfrom [path/to/trained/model.pth] --folder [optional/path/to/folder/of/images] --image [optional/path/to/single/image]

To regenerate the train/validation/test splits:

Edit train_test_val_split.py with the desired split proportions, and then run

python train_test_val_split.py

To regenerate the list of classes, run

python generate_class_file.py --path [path/to/a/folder/of/species/folders] --classfile [path/to/write/new/classfile.txt]


Drive link to trained models: https://drive.google.com/drive/folders/1GEfVeBjX-W49VwwYMdMwI6XOBEhAHcWQ?usp=sharing

On Drive:

alexnet_100.pth : A model trained for 100 epochs on 80% of leafsnap, validated on 10%, tested on another 10%

alexnet_100_best.pth : The model with the lowest validation loss (but not the highest accuracy) during training (from epoch 16)

trainlogs/alexnet_100.txt : Log of the training/testing losses/accuracies
