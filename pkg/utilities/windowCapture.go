package utilities

import (
	"fmt"
	"image"
	"image/png"
	"os"
	"reflect"
	"syscall"
	"unsafe"

	"github.com/disintegration/gift"

	"wz.com/self-learning/pkg/apis/utilities"
)

func Capture(appName string) error {
	utilities.ProcSetProcessDpiAwareness.Call(uintptr(2)) // PROCESS_PER_MONITOR_DPI_AWARE

	// Find the window
	handle, err := findWindow(appName)
	if err != nil {
		return nil
	}

	// Determine the full width and height of the window
	rect, err := windowRect(handle)
	if err != nil {
		return nil
	}

	buff, _ := captureWindow(handle, rect)
	file, err := os.Create("image.png")
	if err != nil {
		return fmt.Errorf("error encoding image: %s", err)
	}
	defer file.Close()
	if err = png.Encode(file, buff); err != nil {
		return fmt.Errorf("error encoding image: %s", err)
	}
	return nil
}

// Windows RECT structure
type win_RECT struct {
	Left, Top, Right, Bottom int32
}

// http://msdn.microsoft.com/en-us/library/windows/desktop/dd183375.aspx
type win_BITMAPINFO struct {
	BmiHeader *utilities.Win_BITMAPINFOHEADER
	BmiColors *utilities.Win_RGBQUAD
}

// findWindow finds the handle to the window.
func findWindow(appName string) (syscall.Handle, error) {
	var handle syscall.Handle

	// First look for the normal window
	utf16, _ := syscall.UTF16PtrFromString(appName)
	ret, _, _ := utilities.ProcFindWindow.Call(0, uintptr(unsafe.Pointer(utf16)))
	if ret == 0 {
		return handle, fmt.Errorf("App not found. Is it running?")
	}

	handle = syscall.Handle(ret)
	return handle, nil
}

// windowRect gets the dimensions for a Window handle.
func windowRect(hwnd syscall.Handle) (image.Rectangle, error) {
	var rect win_RECT
	ret, _, err := utilities.ProcGetClientRect.Call(uintptr(hwnd), uintptr(unsafe.Pointer(&rect)))
	if ret == 0 {
		return image.Rectangle{}, fmt.Errorf("Error getting window dimensions: %s", err)
	}

	return image.Rect(0, 0, int(rect.Right), int(rect.Bottom)), nil
}

// captureWindow captures the desired area from a Window and returns an image.
func captureWindow(handle syscall.Handle, rect image.Rectangle) (image.Image, error) {
	// Get the device context for screenshotting
	dcSrc, _, err := utilities.ProcGetDC.Call(uintptr(handle))
	if dcSrc == 0 {
		return nil, fmt.Errorf("Error preparing screen capture: %s", err)
	}
	defer utilities.ProcReleaseDC.Call(0, dcSrc)

	// Grab a compatible DC for drawing
	dcDst, _, err := utilities.ProcCreateCompatibleDC.Call(dcSrc)
	if dcDst == 0 {
		return nil, fmt.Errorf("Error creating DC for drawing: %s", err)
	}
	defer utilities.ProcDeleteDC.Call(dcDst)

	// Determine the width/height of our capture
	width := rect.Dx()
	height := rect.Dy()

	// Get the bitmap we're going to draw onto
	var bitmapInfo win_BITMAPINFO
	bitmapInfo.BmiHeader = &utilities.Win_BITMAPINFOHEADER{
		BiSize:        uint32(reflect.TypeOf(bitmapInfo.BmiHeader).Size()),
		BiWidth:       int32(width),
		BiHeight:      int32(height),
		BiPlanes:      1,
		BiBitCount:    32,
		BiCompression: 0, // BI_RGB
	}
	bitmapData := unsafe.Pointer(uintptr(0))
	bitmap, _, err := utilities.ProcCreateDIBSection.Call(
		dcDst,
		uintptr(unsafe.Pointer(&bitmapInfo)),
		0,
		uintptr(unsafe.Pointer(&bitmapData)), 0, 0)
	if bitmap == 0 {
		return nil, fmt.Errorf("Error creating bitmap for screen capture: %s", err)
	}
	defer utilities.ProcDeleteObject.Call(bitmap)

	// Select the object and paint it
	utilities.ProcSelectObject.Call(dcDst, bitmap)
	ret, _, err := utilities.ProcBitBlt.Call(
		dcDst, 0, 0, uintptr(width), uintptr(height),
		dcSrc, uintptr(rect.Min.X), uintptr(rect.Min.Y), utilities.BitBlt_SRCCOPY)
	if ret == 0 {
		return nil, fmt.Errorf("Error capturing screen: %s", err)
	}

	// Convert the bitmap to an image.Image. We first start by directly
	// creating a slice. This is unsafe but we know the underlying structure
	// directly.
	var slice []byte
	sliceHdr := (*reflect.SliceHeader)(unsafe.Pointer(&slice))
	sliceHdr.Data = uintptr(bitmapData)
	sliceHdr.Len = width * height * 4
	sliceHdr.Cap = sliceHdr.Len

	// Using the raw data, grab the RGBA data and transform it into an image.RGBA
	imageBytes := make([]byte, len(slice))
	for i := 0; i < len(imageBytes); i += 4 {
		imageBytes[i], imageBytes[i+2], imageBytes[i+1], imageBytes[i+3] = slice[i+2], slice[i], slice[i+1], slice[i+3]
	}

	// The window gets screenshotted upside down and I don't know why.
	// Flip it.
	img := &image.RGBA{Pix: imageBytes, Stride: 4 * width, Rect: image.Rect(0, 0, width, height)}
	dst := image.NewRGBA(img.Bounds())
	gift.New(gift.FlipVertical()).Draw(dst, img)

	return dst, nil
}
