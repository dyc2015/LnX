// See https://aka.ms/new-console-template for more information
using Lnx.ConsoleApp;
using System.Diagnostics;
using System.Text.Json;

//Console.WriteLine("Hello, World!");
//Console.WriteLine(typeof(string).Assembly.FullName);

var stopwatch = new Stopwatch();
stopwatch.Start();

Console.WriteLine(LeeCode.LetterCombinations());

stopwatch.Stop();


Console.WriteLine(stopwatch.ElapsedMilliseconds);