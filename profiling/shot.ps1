# Captures the game window of a best-run/boot-run tag to a PNG (brings it to the foreground first).
param([Parameter(Mandatory)][string]$Tag, [Parameter(Mandatory)][string]$OutFile, [double]$Scale = 0.5)
$ErrorActionPreference = 'Stop'
$identity = [System.Text.Json.JsonDocument]::Parse([System.IO.File]::ReadAllText("D:\ps5\gt7\diagnostics-rapid\$Tag\process.json")).RootElement
$process = Get-Process -Id $identity.GetProperty('pid').GetInt32() -ErrorAction Stop
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class GameShot {
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
    [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
    [DllImport("user32.dll")] public static extern void keybd_event(byte k, byte s, uint f, UIntPtr e);
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
}
'@
[void][GameShot]::SetProcessDPIAware()
$handle = $process.MainWindowHandle
if ($handle -eq [IntPtr]::Zero) { throw "no main window" }
[void][GameShot]::ShowWindow($handle, 9)
[GameShot]::keybd_event(0x12, 0, 0, [UIntPtr]::Zero); [GameShot]::keybd_event(0x12, 0, 2, [UIntPtr]::Zero)
[void][GameShot]::SetForegroundWindow($handle)
Start-Sleep -Milliseconds 300
$rect = New-Object GameShot+RECT
[void][GameShot]::GetClientRect($handle, [ref]$rect)
$origin = New-Object GameShot+POINT
[void][GameShot]::ClientToScreen($handle, [ref]$origin)
$w = $rect.Right - $rect.Left; $h = $rect.Bottom - $rect.Top
$bmp = New-Object System.Drawing.Bitmap $w, $h
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($origin.X, $origin.Y, 0, 0, (New-Object System.Drawing.Size $w, $h))
$sw = [int]($w * $Scale); $sh = [int]($h * $Scale)
$small = New-Object System.Drawing.Bitmap $bmp, $sw, $sh
$small.Save($OutFile, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose(); $small.Dispose()
# Mean brightness, so scripts can tell a black screen without looking.
$probe = New-Object System.Drawing.Bitmap $OutFile
$sum = 0; $n = 0
for ($y = 0; $y -lt $probe.Height; $y += 8) { for ($x = 0; $x -lt $probe.Width; $x += 8) { $c = $probe.GetPixel($x, $y); $sum += ($c.R + $c.G + $c.B) / 3; $n++ } }
$probe.Dispose()
Write-Output ("saved {0} {1}x{2} mean {3:N1}" -f $OutFile, $sw, $sh, ($sum / [math]::Max($n, 1)))
