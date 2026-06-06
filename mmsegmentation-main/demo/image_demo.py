import cv2
import numpy as np
from mmseg.apis import init_model, inference_segmentor, show_result_pyplot
from torch.utils.tensorboard import SummaryWriter  # 引入TensorBoard Writer

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--opacity', type=float, default=0.5, help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--with-labels', action='store_true', default=False, help='Whether to display the class labels.')
    parser.add_argument('--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # 创建 TensorBoard Writer
    writer = SummaryWriter('./work_dirs/segformer_b2_uavid/tensorboard_logs')  # 指定TensorBoard日志保存路径

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    
    # test a single image
    result = inference_segmentor(model, args.img)

    # 保存预测图
    result_file = args.out_file if args.out_file else 'predicted_image.png'
    show_result_pyplot(model, args.img, result, title=args.title, opacity=args.opacity, with_labels=args.with_labels, draw_gt=False, show=False, out_file=result_file)

    # 将预测图保存到 TensorBoard
    result_image = np.array(result[0])  # 将结果转换为 numpy 数组
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)  # 转换颜色格式为 RGB

    # 将预测图写入 TensorBoard
    writer.add_image('Predicted Image', result_image, global_step=0)

    # 关闭 TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()