using LnX.ML.Utils;
using Microsoft.Win32;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;

namespace LnX.ML
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void DNNBtn_Click(object sender, RoutedEventArgs e)
        {
            var window = new DNNWindow();
            window.ShowDialog();
        }

        private void CNNBtn_Click(object sender, RoutedEventArgs e)
        {
            var window = new CNNWindow();
            window.ShowDialog();

        }
    }
}