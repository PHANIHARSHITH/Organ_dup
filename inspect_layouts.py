from pptx import Presentation
prs = Presentation('final_ppt.pptx')
print('Number of layouts', len(prs.slide_layouts))
for i, layout in enumerate(prs.slide_layouts):
    print(i, layout.name)
