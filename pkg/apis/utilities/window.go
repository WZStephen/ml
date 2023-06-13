package utilities

import "syscall"

const (
	SW_SHOWMINIMIZED = 2
	GW_HWNDNEXT      = 2
	// GetDeviceCaps constants from Wingdi.h
	deviceCaps_HORZRES    = 8
	deviceCaps_VERTRES    = 10
	deviceCaps_LOGPIXELSX = 88
	deviceCaps_LOGPIXELSY = 90
	BitBlt_SRCCOPY        = 0x00CC0020 // BitBlt constants
)

type RECT struct {
	Left   int32
	Top    int32
	Right  int32
	Bottom int32
}

var (
	user32                   = syscall.MustLoadDLL("user32.dll")
	ProcEnumWindows          = user32.MustFindProc("EnumWindows")
	ProcGetWindowTextW       = user32.MustFindProc("GetWindowTextW")
	FindWindow               = user32.MustFindProc("FindWindowW")
	getWindowRect            = user32.MustFindProc("GetWindowRect")
	getWindowPlacement       = user32.MustFindProc("GetWindowPlacement")
	setForegroundWindow      = user32.MustFindProc("SetForegroundWindow")
	setWindowPos             = user32.MustFindProc("SetWindowPos")
	showWindow               = user32.MustFindProc("ShowWindow")
	getWindowThreadProcessId = user32.MustFindProc("GetWindowThreadProcessId")

	modUser32                  = syscall.NewLazyDLL("User32.dll")
	ProcFindWindow             = modUser32.NewProc("FindWindowW")
	ProcGetClientRect          = modUser32.NewProc("GetClientRect")
	ProcGetDC                  = modUser32.NewProc("GetDC")
	ProcReleaseDC              = modUser32.NewProc("ReleaseDC")
	modGdi32                   = syscall.NewLazyDLL("Gdi32.dll")
	ProcBitBlt                 = modGdi32.NewProc("BitBlt")
	ProcCreateCompatibleBitmap = modGdi32.NewProc("CreateCompatibleBitmap")
	ProcCreateCompatibleDC     = modGdi32.NewProc("CreateCompatibleDC")
	ProcCreateDIBSection       = modGdi32.NewProc("CreateDIBSection")
	ProcDeleteDC               = modGdi32.NewProc("DeleteDC")
	ProcDeleteObject           = modGdi32.NewProc("DeleteObject")
	ProcGetDeviceCaps          = modGdi32.NewProc("GetDeviceCaps")
	ProcSelectObject           = modGdi32.NewProc("SelectObject")
	modShcore                  = syscall.NewLazyDLL("Shcore.dll")
	ProcSetProcessDpiAwareness = modShcore.NewProc("SetProcessDpiAwareness")
)

// http://msdn.microsoft.com/en-us/library/windows/desktop/dd162938.aspx
type Win_RGBQUAD struct {
	RgbBlue     byte
	RgbGreen    byte
	RgbRed      byte
	RgbReserved byte
}

// http://msdn.microsoft.com/en-us/library/windows/desktop/dd183376.aspx
type Win_BITMAPINFOHEADER struct {
	BiSize          uint32
	BiWidth         int32
	BiHeight        int32
	BiPlanes        uint16
	BiBitCount      uint16
	BiCompression   uint32
	BiSizeImage     uint32
	BiXPelsPerMeter int32
	BiYPelsPerMeter int32
	BiClrUsed       uint32
	BiClrImportant  uint32
}
