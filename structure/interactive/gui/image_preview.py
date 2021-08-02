""" Handles displaying the image preview """
import wx


class ImagePreview(wx.Panel):
    def __init__(self, parent: wx.Window=None):
        """ Creates an image preview with the specified parent window """
        super(ImagePreview, self).__init__(parent, id=wx.ID_ANY)
        self.parent = parent
        self.bitmap = None
        self.draw_bitmap = None
        self.image = None
        self.background_color = wx.BLACK_BRUSH
        width, height = self.GetClientSize()
        self.buffer = wx.Bitmap(width, height)

        # load the background color from the parent
        if parent is not None:
            self.background_color = wx.Brush(parent.GetBackgroundColour())

        self.SetDoubleBuffered(True)
        self.update_bitmap()
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_resize)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase)

    def update_bitmap(self):
        """ Updates the bitmap or image data to fit the screen """
        if self.image is not None:  # there is a valid image
            image = self.image
        elif self.bitmap is not None:  # there was only a bitmap provided
            image = self.bitmap.ConvertToImage()
        else:
            image = None

        # redraw the buffer
        dc = wx.MemoryDC(self.buffer)
        dc.SelectObject(self.buffer)
        dc.SetBackground(self.background_color)
        dc.Clear()

        # draw the image if it's valid
        if image is not None:
            # scale the image to the client dimensions and try to contain the aspect ratio
            width, height = image.GetWidth(), image.GetHeight()
            client_width, client_height = self.GetClientSize()

            # resize with aspect ratio
            if client_width < width or client_height < height:
                scale = float(client_width) / width

                if scale > float(client_height) / height:
                    scale = float(client_height) / height

                new_w, new_h = int(width * scale), int(height * scale)
                if new_w < 1:
                    new_w = 1
                if new_h < 1:
                    new_h = 1

                image = image.Scale(new_w, new_h, wx.IMAGE_QUALITY_HIGH)

            # draw the bitmap in the center of the screen
            self.draw_bitmap = image.ConvertToBitmap()
            self.center_coords = int((client_width - self.draw_bitmap.GetWidth()) // 2), int((client_height - self.draw_bitmap.GetHeight()) // 2)
            dc.DrawBitmap(self.draw_bitmap, *self.center_coords)

        del dc  # push the updates to the buffer

        # push the buffer to the screen
        self.Refresh(False)

    def set_bitmap(self, bitmap):
        """ Updates the bitmap

        :param bitmap: the wx.Bitmap or wx.Image object
        """
        if isinstance(bitmap, wx.Image):
            self.image = bitmap
            self.bitmap = None
        else:
            self.bitmap = bitmap
            self.image = None
        self.update_bitmap()

    def set_background_color(self, color):
        """ Set the background color of the area that isn't exposed by the image

        :param color: the wx Color object
        """
        self.background_color = wx.Brush(color)
        self.update_bitmap()

    def on_erase(self, event):
        """ Handle the erase background event """
        pass

    def on_resize(self, event):
        """ Handles the resizing of the window """
        width, height = self.GetClientSize()
        self.buffer = wx.Bitmap(width, height)
        self.update_bitmap()

    def on_paint(self, event):
        """ Periodic paint event """
        wx.BufferedPaintDC(self, self.buffer)
