using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Media3D;

namespace LnX.ML.CNN
{
    /// <summary>
    /// 卷积神经网络
    /// </summary>
    public class ConvolutionalNeuralNetwork
    {
        /// <summary>
        /// 创建一个卷积变换
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="stride"></param>
        /// <returns></returns>
        public static ConvolutionalTransformer ConvolutionalTransformer(Tensor kernel, IDifferentiableFunction<double> function, int stride = 1)
        {
            return new ConvolutionalTransformer(kernel, function, stride);
        }
    }

    public interface INeuron
    {

    }
}
