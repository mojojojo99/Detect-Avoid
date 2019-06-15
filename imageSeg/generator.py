train_image_generator = train_datagen.flow_from_directory(
'data/train_frames/train',
batch_size = #NORMALLY 4/8/16/32)

train_mask_generator = train_datagen.flow_from_directory(
'data/train_masks/train',
batch_size = #NORMALLY 4/8/16/32)

val_image_generator = val_datagen.flow_from_directory(
'data/val_frames/val',
batch_size = #NORMALLY 4/8/16/32)


val_mask_generator = val_datagen.flow_from_directory(
'data/val_masks/val',
batch_size = #NORMALLY 4/8/16/32)



train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)

