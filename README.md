# Painterly-Style-Transfer

This is the code for the paper

**[Painterly Style Transfer with Learned Brush Strokes](https://ieeexplore.ieee.org/document/10319299)**

Accepted by IEEE Transactions on Visualization and Computer Graphics

If you find this code useful for your research, please cite

```
@ARTICLE{liu2023PainterlyST,
	author={Liu, Xiao-Chang and Wu, Yu-Chen and Hall, Peter},
	journal={IEEE Transactions on Visualization and Computer Graphics},
	title={Painterly Style Transfer with Learned Brush Strokes},
	year={2023},
	doi={10.1109/TVCG.2023.3332950}
}
```

## Preresquisites

pip3 install --r requirements.txt

## Running on test images

```bash
python3 plan.py \
  --objective_data nst_pixel/bridge_nst.png \
  --objective clip_conv_loss \
  --objective_weight 1.0 \
  --optim_iter 400 \
  --stroke_length 0.3 \
  --stroke_curva 0.1 \
  --max_height 300 \
  --num_strokes 200 \
  --base_canvas init_placement/bridge_init.jpg \
  --middle_result_name bridge_strokes.jpg
```
## Acknowledgement
This project is inspired by many existing methods and their open-source implementations, including:

