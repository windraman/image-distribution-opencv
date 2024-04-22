import datetime
import pathlib
from queue import Queue
from threading import Thread
from tkinter.filedialog import askdirectory,askopenfilename 
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap import utility
import cv2
from PIL import Image,ImageTk,ImageGrab
import numpy as np
from skimage import measure
import tkinter as tk 
from tkinter import filedialog

def new_canvas(h,w):
    b, g, r = 255, 255, 255  # orange
    new_img = np.zeros((h, w, 3), np.uint8)
    new_img[:, :, 0] = b
    new_img[:, :, 1] = g
    new_img[:, :, 2] = r

    return new_img


def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

    return background

def find_contour(image):
    median = np.median(image) 
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    
    edge_image= cv2.Canny(image, lower, upper)
    

    contours, hierarchy = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    # print(contours)
    # if(contours):
    #     contours = contours[1] if len(contours) == 3 else contours[0]

    return contours

def check_intersection(polygon1, polygon2):
    intersection = False
    for point in polygon2:
        result = cv2.pointPolygonTest(polygon1, tuple(point), measureDist=False)
        # if point inside return 1
        # if point outside return -1
        # if point on the contour return 0

        if result == 1:
            intersection = True

    return intersection

def detect_area(image, invert=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    if(invert):
        thresh = cv2.bitwise_not(thresh)

    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    labels = measure.label(thresh, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 300:
            mask = cv2.add(mask, labelMask)
        
    return mask

def rotateImage(image, angle):
    center=tuple(np.array(image.shape[0:2])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[0:2],flags=cv2.INTER_LINEAR)

class FileSearchEngine(ttk.Frame):

    queue = Queue()
    searching = False

    def __init__(self, master):
        super().__init__(master, padding=15)
        self.pack(fill=BOTH, expand=YES)

        # application variables
        _path = pathlib.Path().absolute().as_posix()
        self.canvas_x = ttk.IntVar(value=1024)
        self.canvas_y = ttk.IntVar(value=720)
        self.file_var = ttk.StringVar(value="")
        self.file_num_var = ttk.IntVar(value=10)
        self.file_margin_var = ttk.IntVar(value=50)
        self.file_pattern_var = ttk.StringVar(value="0,180")
        self.path_var = ttk.StringVar(value=_path)
        self.term_var = ttk.StringVar(value='md')
        self.type_var = ttk.StringVar(value='endswidth')
        self.images = [["no image",0]]

        option_canvas_text = "Canvas Settings"
        self.option_canvas = ttk.Labelframe(self, text=option_canvas_text, padding=15)
        self.option_canvas.pack(fill=X, expand=YES, anchor=N)

       

        # header and labelframe option container
        option_text = "Add PNG file to begin"
        self.option_lf = ttk.Labelframe(self, text=option_text, padding=15)
        self.option_lf.pack(fill=X, expand=YES, anchor=N)

        self.create_canvas_row()

        self.create_file_row()
        self.create_file_number_row()
        self.create_file_margin_row()
        self.create_file_pattern_row()
        self.create_type_row()

        self.create_results_view()

        option_gen_text = ""
        self.option_gen = ttk.Labelframe(self, text=option_gen_text, padding=15)
        self.option_gen.pack(fill=X, expand=YES, anchor=N)

        self.create_gen_row()

    def create_canvas_row(self):
        """Add path row to labelframe"""
        path_row = ttk.Frame(self.option_canvas)
        path_row.pack(fill=X, expand=YES)
        path_lbl = ttk.Label(path_row, text="Canvas Size", width=15)
        path_lbl.pack(side=LEFT, padx=(15, 0))
        path_ent = ttk.Entry(path_row, textvariable=self.canvas_x)
        path_ent.pack(side=LEFT, fill=X, expand=YES, padx=5)
        path_ent = ttk.Entry(path_row, textvariable=self.canvas_y)
        path_ent.pack(side=LEFT, fill=X, expand=YES, padx=5)
      

    def create_file_row(self):
        """Add path row to labelframe"""
        path_row = ttk.Frame(self.option_lf)
        path_row.pack(fill=X, expand=YES)
        path_lbl = ttk.Label(path_row, text="PNG File", width=15)
        path_lbl.pack(side=LEFT, padx=(15, 0))
        path_ent = ttk.Entry(path_row, textvariable=self.file_var)
        path_ent.pack(side=LEFT, fill=X, expand=YES, padx=5)
        browse_btn = ttk.Button(
            master=path_row, 
            text="Browse", 
            command=self.on_browse_file, 
            width=8
        )
        browse_btn.pack(side=LEFT, padx=5)

    def create_file_number_row(self):
        """Add path row to labelframe"""
        path_row = ttk.Frame(self.option_lf)
        path_row.pack(fill=X, expand=YES)
        path_lbl = ttk.Label(path_row, text="Quantity", width=15)
        path_lbl.pack(side=LEFT, padx=(15, 0))
        path_ent = ttk.Entry(path_row, textvariable=self.file_num_var, width=5)
        path_ent.pack(side=LEFT, fill=X, expand=YES, padx=5, pady=5)

    
    def create_file_margin_row(self):
        """Add path row to labelframe"""
        path_row = ttk.Frame(self.option_lf)
        path_row.pack(fill=X, expand=YES)
        path_lbl = ttk.Label(path_row, text="Step", width=15)
        path_lbl.pack(side=LEFT, padx=(15, 0))
        path_ent = ttk.Entry(path_row, textvariable=self.file_margin_var, width=9)
        path_ent.pack(side=LEFT, fill=X, expand=YES, padx=5, pady=5)
    
    def create_file_pattern_row(self):
        """Add path row to labelframe"""
        path_row = ttk.Frame(self.option_lf)
        path_row.pack(fill=X, expand=YES)
        path_lbl = ttk.Label(path_row, text="Rotate Pattern", width=15)
        path_lbl.pack(side=LEFT, padx=(15, 0))
        path_ent = ttk.Entry(path_row, textvariable=self.file_pattern_var, width=9)
        path_ent.pack(side=LEFT, fill=X, expand=YES, padx=5, pady=5)
        browse_btn = ttk.Button(
            master=path_row, 
            text="Add", 
            command=self.on_add_file, 
            width=8
        )
        browse_btn.pack(side=LEFT, padx=5)

    def create_generate_row(self):
        search_btn = ttk.Button(
            master=self  , 
            text="Generate", 
            command=self.on_generate, 
            bootstyle=OUTLINE, 
            width=8
        )
        search_btn.pack(side=LEFT, padx=5)

    def create_gen_row(self):
        gen_row = ttk.Frame(self.option_gen)
        gen_row.pack(fill=X, expand=YES)
        search_btn = ttk.Button(
            master=gen_row  , 
            text="Generate", 
            command=self.on_generate, 
            bootstyle=OUTLINE, 
            width=8
        )
        search_btn.pack(side=LEFT, padx=5)

        self.progressbar = ttk.Progressbar(
            master=gen_row, 
            mode=INDETERMINATE, 
            bootstyle=(STRIPED, SUCCESS)
        )
        self.progressbar.pack(side=LEFT, fill=X, expand=YES, padx=5, pady=5)

        save_btn = ttk.Button(
            master=gen_row  , 
            text="Save", 
            command=self.save_as_png, 
            bootstyle=OUTLINE, 
            width=8
        )
        save_btn.pack(side=LEFT, padx=5)


    def create_type_row(self):
        """Add type row to labelframe"""
        type_row = ttk.Frame(self.option_lf)
        type_row.pack(fill=X, expand=YES)
        type_lbl = ttk.Label(type_row, text="Metode", width=15)
        type_lbl.pack(side=LEFT, padx=(15, 0))

        contains_opt = ttk.Radiobutton(
            master=type_row, 
            text="Linear", 
            variable=self.type_var, 
            value="linear"
        )
        contains_opt.pack(side=LEFT)

        startswith_opt = ttk.Radiobutton(
            master=type_row, 
            text="Spirial", 
            variable=self.type_var, 
            value="spirial"
        )
        startswith_opt.pack(side=LEFT, padx=15)

        contains_opt.invoke()

    def create_results_view(self):
        """Add result treeview to labelframe"""
        self.resultview = ttk.Treeview(
            master=self, 
            bootstyle=INFO, 
            columns=[0, 1, 2, 3, 4, 5],
            show=HEADINGS
        )
        self.resultview.pack(fill=BOTH, expand=YES, pady=10)

        # setup columns and use `scale_size` to adjust for resolution
        self.resultview.heading(0, text='Name', anchor=W)
        self.resultview.heading(1, text='Size', anchor=E)
        self.resultview.heading(2, text='Qty', anchor=E)
        self.resultview.heading(3, text='Path', anchor=W)
        self.resultview.heading(4, text='Step', anchor=E)
        self.resultview.heading(5, text='Pattern', anchor=E)
        self.resultview.column(
            column=0, 
            anchor=W, 
            width=utility.scale_size(self, 100), 
            stretch=False
        )
        self.resultview.column(
            column=1, 
            anchor=E, 
            width=utility.scale_size(self, 50), 
            stretch=False
        )
        self.resultview.column(
            column=2, 
            anchor=E, 
            width=utility.scale_size(self, 50), 
            stretch=False
        )
        self.resultview.column(
            column=3, 
            anchor=W, 
            width=utility.scale_size(self, 300), 
            stretch=False
        )
        self.resultview.column(
            column=4, 
            anchor=E, 
            width=utility.scale_size(self, 60), 
            stretch=False
        )
        self.resultview.column(
            column=5, 
            anchor=E, 
            width=utility.scale_size(self, 70), 
            stretch=False
        )

        self.resultview.bind("<Button-3>", self.popup)
        # edit_btn = ttk.Button(app, text="Edit", command=self.edit)
        # edit_btn.pack()
        # del_btn = ttk.Button(app, text="Delete", command=self.delete)
        # del_btn.pack()

    def popup(self, event):
        """action in event of button 3 on tree view"""
        # select row under mouse
        iid = self.resultview.identify_row(event.y)
        if iid:
            self.right_click_menu = tk.Menu(self, bg="lightgrey", fg="black", tearoff=0)
            self.right_click_menu.add_command(label='Edit', command=self.edit)
            self.right_click_menu.add_command(label='Delete', command=self.delete)
            self.resultview.selection_set(iid)
            self.right_click_menu.post(event.x_root, event.y_root)            
        else:
            # mouse pointer not over item
            # occurs when items do not fill frame
            # no action required
            pass

    def on_add_file(self):
        if(self.images[0][0]=="no image"):
            self.images = []
            self.images.append([self.file_var.get(),self.file_num_var.get(),self.file_margin_var.get(),self.file_pattern_var.get().split(","),self.type_var.get()])
        else:
            self.images.append([self.file_var.get(),self.file_num_var.get(),self.file_margin_var.get(),self.file_pattern_var.get().split(","),self.type_var.get()])

        self.insert_image_row([self.file_var.get(),self.file_num_var.get(),self.file_margin_var.get(),self.file_pattern_var.get().split(","),self.type_var.get()])
        print(self.images)
    
    def edit(self):
        # Get selected item to Edit
        selected_item = self.resultview.selection()[0]
        self.resultview.item(selected_item, text="blub", values=("foo", "bar"))

    def delete(self):
        # Get selected item to Delete
        selected_item = self.resultview.selection()[0]
        self.resultview.delete(selected_item)
        # self.images.remove(self.resultview.selection())

    def on_browse_file(self):
        """Callback for directory browse"""
        path = askopenfilename (title="Browse file PNG")
        if path:
            self.file_var.set(path)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            height, width = img.shape[:2]
            self.file_margin_var.set(int(width/3))

    def on_generate(self):
        second = tk.Toplevel()
        second.title("Generate") # Rename this
        second.geometry(str(self.canvas_x.get())+"x"+str(self.canvas_y.get()))
        self.canvas =tk.Canvas(second,height=self.canvas_y.get(),width=self.canvas_x.get(), background="white")
        self.canvas.pack()
    
        second.img_ref = []
        
        self.gen_contours = []
        self.index = 0

        self.cy,self.cx = self.canvas_y.get(),self.canvas_x.get()
        self.centery,self.centerx = self.cy/2,self.cx/2
        b, g, r = 255, 255, 255  # orange
        self.base = np.zeros((self.cy, self.cx, 3), np.uint8)
        self.base[:, :, 0] = b
        self.base[:, :, 1] = g
        self.base[:, :, 2] = r

        self.total = 0
        for item in self.images:
            self.total += item[1]

        

        # create a canvas

        # this data is used to keep track of an
        # item being dragged
        self._drag_data = {"x": 0, "y": 0, "item": None}
        self.progressbar.start(10)
        for item in self.images:
            name = item[0]
            print('processing', name)
            
            added = 0
            y,x,d = 0,0,0

            self.step = item[2]
        
            self.derajat = item[3]
            self.max_rot = len(self.derajat)
            ms = 1
            for i in range(0, item[1]):  
                mask_combined = False
                m = 0
                canvas_area = self.base.copy()
                canvas_contour = find_contour(canvas_area)
                
                # if(len(canvas_contour)==0):
                #     print("no contour")     
                while(mask_combined==False):
                    img = cv2.imread(name, cv2.IMREAD_UNCHANGED)
                    height, width = img.shape[:2]
                    newbase = new_canvas(self.cy,self.cx)
                    base = cv2.drawContours(newbase, canvas_contour, -1, (0, 255, 0), thickness=cv2.FILLED) 
                    if(item[4]=="linear"):
                        if(d>=self.max_rot):
                            d = 0
                            x += self.step
                            if(x + width>=self.cx):
                                x = 0
                                y += self.step
                                if(y + height>=self.cy):
                                    y = 0
                    
                    if(item[4]=="spirial"):
                        if(d>=self.max_rot):
                            d = 0
                            x += self.step
                            if(x >= width * ms - 10):
                                x = 0
                                y += self.step
                                if(y >= height * ms - 10):
                                    y = 0
                                    ms += 1
                    
                    new_point = [int(y),int(x),int(self.derajat[d])]
                    # print(new_point)
                    d += 1

                    if(new_point[0]+height<=self.cy and new_point[1]+width<=self.cx):
                        newc = new_canvas(self.cy,self.cx)
                        img = rotateImage(img, new_point[2])
                        newc = add_transparent_image(newc,img,new_point[1],new_point[0])
                        
                        img_contour = find_contour(newc)  
                        oview = new_canvas(self.cy,self.cx) 
                        overlay =cv2.drawContours(oview, img_contour, -1, (0, 255, 0), thickness=cv2.FILLED) 


                        combined = cv2.bitwise_or(base, overlay)
                        mask_combined = combined.all()

                        
                        if (mask_combined==True):
                            print(new_point[0],new_point[1],new_point[2])
                            self.base = add_transparent_image(self.base,img,new_point[1],new_point[0])
                            self.second = second
                            self.create_dimage(new_point[1], new_point[0], img)

                            self.canvas.tag_bind("token", "<ButtonPress-1>", self.drag_start)
                            self.canvas.tag_bind("token", "<ButtonRelease-1>", self.drag_stop)
                            self.canvas.tag_bind("token", "<B1-Motion>", self.drag)
                            self.canvas.tag_bind("token", "<Button-3>", self.right_click)

                            added += 1
                            print(added, name,"added")
                            second.update()
                            
                        

                    if(x + width>=self.cx and y + height>=self.cy):
                        # print("limit", + width,contourw,y + height,contourh)
                        self.progressbar.stop()
                        break
                        # mask_combined = True
                    
        self.progressbar.stop()
                
    

    def save_as_png(self):
        save_name= filedialog.asksaveasfilename()
        self.second.lift()
        x=self.canvas.winfo_rootx()+self.canvas.winfo_x()
        y=self.canvas.winfo_rooty()+self.canvas.winfo_y()
        x1=x+self.canvas.winfo_width()
        y1=y+self.canvas.winfo_height()
        ImageGrab.grab().crop((x,y,x1,y1)).save(save_name+".png")


    def create_dimage(self, x, y,img):
        """Create a token at    the given coordinate in the given color"""
        color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        pil_image = Image.fromarray(color_coverted) 
        self.filename = ImageTk.PhotoImage(pil_image)
        self.second.filename = self.filename
        self.canvas.create_image(
            x,
            y,
            anchor=NW,
            image=self.filename,
            tags=("token",),
        )

        self.second.img_ref.append(self.filename)

    def drag_start(self, event):
        """Begining drag of an object"""
        # record the item and its location
        self._drag_data["item"] = self.canvas.find_closest(event.x, event.y)[0]
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def drag_stop(self, event):
        """End drag of an object"""
        # reset the drag information
        self._drag_data["item"] = None
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0

    def drag(self, event):
        """Handle dragging of an object"""
        # compute how much the mouse has moved
        delta_x = event.x - self._drag_data["x"]
        delta_y = event.y - self._drag_data["y"]
        # move the object the appropriate amount
        self.canvas.move(self._drag_data["item"], delta_x, delta_y)
        # record the new position
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def right_click(self, event):
        print(self.canvas.itemcget("token","image"))

    
    def insert_image_row(self, image):
        """Insert new row in tree search results"""
        img = cv2.imread(image[0], cv2.IMREAD_UNCHANGED)
        height, width = img.shape[:2]
        color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        pil_image = Image.fromarray(color_converted)
        try:
            _name = ImageTk.PhotoImage(pil_image)
            _qty = image[1]
            _size = width, height
            _path = image[0]
            _margin = image[2]
            _pattern = image[3]
            iid = self.resultview.insert(
                parent='', 
                index=END, 
                text=image[0],
                image=ImageTk.PhotoImage(pil_image),
                values=(_name, _size, _qty, _path, _margin, _pattern)
            )
            self.resultview.selection_set(iid)
            self.resultview.see(iid)
        except OSError:
            return   


if __name__ == '__main__':
    sigma = 0.33
    app = ttk.Window("Image Distribution", "journal")
    FileSearchEngine(app)
    app.mainloop()
