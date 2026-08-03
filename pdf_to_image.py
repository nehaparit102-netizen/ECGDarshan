import fitz  # PyMuPDF

pdf_path = 'data/friend_ecg.jpg.pdf'
doc = fitz.open(pdf_path)

page = doc[0]
pix = page.get_pixmap(dpi=200)
pix.save('data/friend_ecg.jpg')

print("Converted! Saved as data/friend_ecg.jpg ✅")
print(f"Image size: {pix.width} x {pix.height}")