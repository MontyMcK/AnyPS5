# Clicks the game window at a point given in the game's 1920x1080 space (APS5_MOUSE_SPACE), after
# bringing the window to the foreground. Usage: click.ps1 -Tag <tag> -X <0..1919> -Y <0..1079>.
param([Parameter(Mandatory)][string]$Tag, [Parameter(Mandatory)][int]$X, [Parameter(Mandatory)][int]$Y, [int]$Width = 1920, [int]$Height = 1080)
$ErrorActionPreference = 'Stop'
$identity = [System.Text.Json.JsonDocument]::Parse([System.IO.File]::ReadAllText("D:\ps5\gt7\diagnostics-rapid\$Tag\process.json")).RootElement
$process = Get-Process -Id $identity.GetProperty('pid').GetInt32() -ErrorAction Stop
if ($process.Path -ne $identity.GetProperty('path').GetString()) { throw "PID does not match $Tag" }
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class GameMouse {
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr handle);
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr handle, out RECT rect);
    [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr handle, ref POINT point);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extra);
}
'@
$handle = $process.MainWindowHandle
[void][GameMouse]::SetForegroundWindow($handle)
Start-Sleep -Milliseconds 300
$rect = New-Object GameMouse+RECT
[void][GameMouse]::GetClientRect($handle, [ref]$rect)
$point = New-Object GameMouse+POINT
$point.X = [int]($X * $rect.Right / $Width)
$point.Y = [int]($Y * $rect.Bottom / $Height)
[void][GameMouse]::ClientToScreen($handle, [ref]$point)
[void][GameMouse]::SetCursorPos($point.X, $point.Y)
Start-Sleep -Milliseconds 400
[GameMouse]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero)
Start-Sleep -Milliseconds 150
[GameMouse]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)
"clicked ($X,$Y) -> screen ($($point.X),$($point.Y))"
