from data_loader import sed_dataset

ds = sed_dataset.Dataset()
train_generator, length = ds.load_data_generator('train')

print(length)
for i,j,k,l in train_generator:
    print(i,j,k,l)

