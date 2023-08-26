# LyCORIS-evaluation

## Encoding and Evaluation

We suppose that we have a dataset directory containing training images and a generated image directory containing generated images for multiple checkpoints each in a separate subfolder. The organizational structure of each subfolder mirrors that of the training dataset.

### Encoding

```
python encode_image_features.py \
    --src_dir /path/to/generated_images \
    --dst_dir /path/to/generated_features \
    --encoder_names {dinov2-l-fb|convnextv2-l|clip-L-14} \
    --resize_mode {crop|padding|resize} \
    --generated --batch_size 8
```

For generated images which are square, just use `--resize_mode resize`. Dataset images are encoded in the same way but without `--generated`. For non-square images and `convnextv2-l` which can take non-square images as input, `resize` can only be used with `--batch_size=1`.

For text encoding of evaluation prompts, we replace trigger word by words that describe well the concept (see `token_mapping_eval.csv`). Run

```
python encode_text_features.py  \
    --eval_replace_file token_mapping_eval.csv \
    --src_dir /path/to/eval_prompts \
    --dst_dir /path/to/ref_features \
    --n_images 100 
```

### Similarity

#### Image Similarity

```
python eval_image_similarity.py \
    --ref_dir /path/to/ref_features \
    --eval_dir /path/to/generated_features/ \
    --encoder_name $encoder \
    --metric_csv /path/to/metrics.csv
```

#### Text Similarity

```
python eval_text_similarity.py \
    --ref_dir path/to/ref_features \ 
    --eval_dir /path/to/generated_features/ \
    --metric_csv /path/to/metrics.csv
```

#### Frechet Distance

Balance features with repetition for dataset and subsampling for generated images
```
python get_balanced_features.py \
    -- /path/to/ref_features \
    --eval_dir /path/to/generated_features
```

Evaluation Frechet Distance
```
python eval_fd.py \
    --ref_features /path/to/ref_features/fd-image-features.npz \
    --eval_dir path/to/generated_features \
    --metric_csv /path/to/metrics.csv
```

### Diversity

```
python eval_image_diversity.py \
    --eval_dir /path/to/generated_features/ \
    --encoder_name $encoder \
    --metric_csv /path/to/metrics.csv
```

### Style loss

#### Compare with dataset
- Encode style features for dataset
    ```
    python encode_image_features.py \
        --src_dir /path/to/dataset/ \
        --dst_dir /path/to/ref_features_vgg \
        --encoder_names vgg19-gram \
        --resize_mode resize \
        --batch_size 1
    ```
- Compute style loss
    ```
    python eval_style_loss.py \
        --ref_dir /path/to/ref_features_vgg/style/ \
        --eval_dir /path/to/generated_images \
        --extra_level 2 \
        --compare_with_dataset \
        --output_csv /path/to/style_losses.csv
    ```
- **Important: we consider only the style subdirectory in ref_features_vgg** (alternatively we can encode only the style part of the dataset and use that directory).

#### Compare with base model

- Encode style features for base model
    ```
    python encode_image_features.py \
        --src_dir /path/to/generated_images/$base_model \
        --dst_dir /path/to/generated_features/$base_model/ \
        --encoder_names vgg19-gram \
        --resize_mode resize \
        --batch_size 8 \
        --style_only
    ```
- Compute style loss
    ```
    python eval_style_loss.py \
        --ref_dir /path/to/generated_features/$base_model/ \
        --eval_dir /path/to/generated_images \
        --save_dir /path/to/generated_features \
        --extra_level 1 \
        --output_csv /path/to/style_losses.csv
    ```
    
### Image qaulity

Note that this may not be very meaningful

```
python eval_image_quality.py \
    --eval_dir /path/to/generated_images \
    --dst_dir /path/to/generated_image_scores/ \
    --scorer_names {liqe|maniqa|artifact} \
    --resize_mode {crop|padding|resize} \
    --batch_size 8 (--generated)
    --metric_csv /path/to/image_quality.csv
```

For non square images, `liqe` and `maniqa` with `resize` should be used with `--batch_size 1`.
