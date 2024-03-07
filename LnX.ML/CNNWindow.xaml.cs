using Microsoft.Win32;
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
    /// CNNWindow.xaml 的交互逻辑
    /// </summary>
    public partial class CNNWindow : Window
    {
        public CNNWindow()
        {
            InitializeComponent();
        }

        /// <summary>
        /// 手写字数据集合
        /// </summary>
        MinstData[] trainData, testData;
        /// <summary>
        /// 手写字流
        /// </summary>
        Stream testDataStream, testLabelStream, trainDataStream, trainLabelStream;

        private void DataFolderPicker_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFolderDialog();

            if (dlg.ShowDialog() ?? false)
            {
                foreach (var fileName in Directory.GetFiles(dlg.FolderName))
                {
                    if (fileName.Contains("t10k-images.idx3-ubyte"))
                    {
                        testDataStream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
                    }

                    if (fileName.Contains("t10k-labels.idx1-ubyte"))
                    {
                        testLabelStream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
                    }

                    if (fileName.Contains("train-images.idx3-ubyte"))
                    {
                        trainDataStream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
                    }

                    if (fileName.Contains("train-labels.idx1-ubyte"))
                    {
                        trainLabelStream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
                    }
                }

                DataFolderPath.Text = dlg.FolderName;
            }
        }

        private void LeanBtn_Click(object sender, RoutedEventArgs e)
        {
            var leanWindow = new LeanWindow
            {
                TestData = testData,
                TrainData = trainData
            };

            leanWindow.ShowDialog();
        }

        private void ReadBtn_Click(object sender, RoutedEventArgs e)
        {
            if (trainData != null) return;

            if (testDataStream == null || testLabelStream == null || trainDataStream == null || trainLabelStream == null)
            {
                MessageBox.Show("数据不完整，情检查！");
                return;
            }

            using (trainDataStream)
            using (trainLabelStream)
                trainData = MinstReader.Read(trainDataStream, trainLabelStream);

            using (testDataStream)
            using (testLabelStream)
                testData = MinstReader.Read(testDataStream, testLabelStream);

            //Img1.Source = ImageSourceUtil.CreateBitmapSource(trainData[0].Pixels)[0];
            //Img2.Source = ImageSourceUtil.CreateBitmapSource(trainData[1].Pixels)[0];
            //Img3.Source = ImageSourceUtil.CreateBitmapSource(trainData[2].Pixels)[0];
            //Img4.Source = ImageSourceUtil.CreateBitmapSource(trainData[3].Pixels)[0];
            //Img5.Source = ImageSourceUtil.CreateBitmapSource(trainData[4].Pixels)[0];
            //Img6.Source = ImageSourceUtil.CreateBitmapSource(testData[5].Pixels)[0];
            //Img7.Source = ImageSourceUtil.CreateBitmapSource(testData[6].Pixels)[0];
            //Img8.Source = ImageSourceUtil.CreateBitmapSource(testData[7].Pixels)[0];
            //Img9.Source = ImageSourceUtil.CreateBitmapSource(testData[8].Pixels)[0];
        }
    }
}
