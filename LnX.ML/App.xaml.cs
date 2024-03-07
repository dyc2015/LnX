using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;
using SkiaSharp;
using System.Configuration;
using System.Data;
using System.Windows;

namespace LnX.ML
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);

            LiveCharts.Configure(config =>
            config
                // you can override the theme 
                // .AddDarkTheme()  

                // In case you need a non-Latin based font, you must register a typeface for SkiaSharp
                .HasGlobalSKTypeface(SKFontManager.Default.MatchCharacter('汉')) // <- Chinese 
                                                                                //.UseRightToLeftSettings() // Enables right to left tooltips 

            // finally register your own mappers
            // you can learn more about mappers at:
            // https://livecharts.dev/docs/wpf/2.0.0-rc2/Overview.Mappers

            // here we use the index as X, and the population as Y 
            // .HasMap<Foo>( .... ) 
            // .HasMap<Bar>( .... ) 
            );
        }
    }

}
