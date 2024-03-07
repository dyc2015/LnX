
namespace Lnx.Remind
{
    public partial class Form1 : Form
    {
        private bool _hidden = false;
        private bool _closed = false;
        private DateTime _lastDate = DateTime.Now;
        private System.Threading.Timer _timer;
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            notifyIcon1.ContextMenuStrip = NotifyContextMenu;
            exit.Click += (s, e) =>
            {
                _closed = true;
                this.Close();
            };

            _timer = new System.Threading.Timer(CheckDate, null, 60000, 0);
        }

        private void CheckDate(object? state)
        {
            if (DateTime.Now.Hour != _lastDate.Hour)
            {
                _lastDate = DateTime.Now;

                //OnNotifyMessage();
            }
        }

        private void notifyIcon1_MouseDoubleClick(object sender, MouseEventArgs e)
        {
            if (_hidden)
            {
                _hidden = false;
                this.Show();
                this.WindowState = FormWindowState.Normal;
                this.Activate();
            }
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (!_closed)
            {
                _hidden = true;
                e.Cancel = true;
                this.Hide();
            }
        }

        private void Form1_FormClosed(object sender, FormClosedEventArgs e)
        {
            if (notifyIcon1 != null)
                notifyIcon1.Dispose();

            if(_timer != null)
                _timer.Dispose();
        }

        private void Form1_Deactivate(object sender, EventArgs e)
        {
            _hidden = true;
        }
    }
}