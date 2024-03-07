using LnX.ML.CNN;
using LnX.ML.Utils;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace LnX.ML
{
    /// <summary>
    /// LeanWindow.xaml 的交互逻辑
    /// </summary>
    public partial class LeanWindow : Window
    {
        /// <summary>
        /// 手写字数据集合
        /// </summary>
        public MinstData[] TrainData, TestData;

        public LeanWindow()
        {
            InitializeComponent();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            Test();
        }

        public void Test()
        {
            var context = new TransformContext(TrainData[0].Pixels, TrainData[0].Labels);
            var transformer = new ConvolutionalTransformer(KernelUtil.Create(4, 4, 3), Function.CreateReLU(), 1);
            transformer.Transform(context);

            //var sources = ImageSourceUtil.CreateBitmapSource(context.Output);
            //TestImg.Source = sources[0];
            //TestImg1.Source = sources[1];
            //TestImg2.Source = sources[2];

            var poolingTransformer = new PoolingTransformer(3, 3, Function.CreateMaxPooling());
            poolingTransformer.Transform(context);

            //sources = ImageSourceUtil.CreateBitmapSource(context.Output);
            //TestImg3.Source = sources[0];
            //TestImg4.Source = sources[1];
            //TestImg5.Source = sources[2];

            var fullyConnectTransformer = new FullyConnectTransformer();
            fullyConnectTransformer.Transform(context);

            var t = context.Output;
        }
    }
}
