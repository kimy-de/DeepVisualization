import argparse
import models
import visualization

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--model_name', default='custom', help='model')
    parser.add_argument('--img_size', default=512, type=int, help='image size')
    parser.add_argument('--layer_num', default=3, type=int, help='layer_num')
    parser.add_argument('--use_l2', default=False, type=bool, help='use L2')
    parser.add_argument('--save_path', default='./pictures/', type=str, help='save path')
    parser.add_argument('--filter_start', default=10, type=int, help='filter start')
    parser.add_argument('--filter_end', default=20, type=int, help='filter end')
    parser.add_argument('--parallel', default=1, type=int, help='parallel')
    parser.add_argument('--step_size', default=0.1, type=float, help='step size')

    args = parser.parse_args()
    print(args)

    model_selection = models.ModelFactory()
    model = model_selection.load_model(args.model_name)

    # Execute
    visualization.show_layer(model, args.layer_num, args.img_size, args.save_path,
                             parallel = args.parallel,
                             filter_start=args.filter_start,
                             filter_end=args.filter_end,
                             step_size= args.step_size,
                             use_L2=args.use_l2)
