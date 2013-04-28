#include <string.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include "viewer.h"

#define FONT_FIRST  ' '

GLubyte font[][12] = {
	{0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
	{0x00,0x00,0x20,0x00,0x20,0x20,0x20,0x20,0x20,0x20,0x00,0x00},
	{0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x50,0x50,0x50,0x00},
	{0x00,0x00,0x00,0x00,0x00,0x50,0xf8,0x50,0xf8,0x50,0x00,0x00},
	{0x00,0x20,0x70,0xa8,0x28,0x28,0x70,0xa0,0xa8,0x70,0x20,0x00},
	{0x00,0x00,0x90,0xa8,0x68,0x30,0x50,0xa8,0xa8,0x78,0x00,0x00},
	{0x00,0x00,0x68,0x90,0xa8,0xa0,0x40,0xa0,0x90,0x60,0x00,0x00},
	{0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x20,0x20,0x20,0x00},
	{0x00,0x10,0x20,0x40,0x40,0x40,0x40,0x40,0x40,0x20,0x10,0x00},
	{0x00,0x40,0x20,0x10,0x10,0x10,0x10,0x10,0x10,0x20,0x40,0x00},
	{0x00,0x00,0x00,0x00,0x00,0x20,0xa8,0x70,0xa8,0x20,0x00,0x00},
	{0x00,0x00,0x00,0x20,0x20,0xf8,0x20,0x20,0x00,0x00,0x00,0x00},
	{0x40,0x20,0x60,0x60,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
	{0x00,0x00,0x00,0x00,0x00,0x70,0x00,0x00,0x00,0x00,0x00,0x00},
	{0x00,0x00,0x30,0x30,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
	{0x80,0x80,0x40,0x40,0x20,0x20,0x10,0x10,0x08,0x08,0x00,0x00},
	{0x00,0x00,0x70,0x88,0x88,0xc8,0xa8,0x98,0x88,0x70,0x00,0x00},
	{0x00,0x00,0xf8,0x20,0x20,0x20,0x20,0x20,0xe0,0x20,0x00,0x00},
	{0x00,0x00,0xf8,0x80,0x40,0x20,0x10,0x08,0x88,0x70,0x00,0x00},
	{0x00,0x00,0x70,0x88,0x08,0x08,0x30,0x08,0x88,0x70,0x00,0x00},
	{0x00,0x00,0x38,0x10,0x10,0xf8,0x90,0x50,0x30,0x10,0x00,0x00},
	{0x00,0x00,0x70,0x88,0x08,0x08,0xf0,0x80,0x80,0xf8,0x00,0x00},
	{0x00,0x00,0x70,0x88,0x88,0x88,0xf0,0x80,0x80,0x70,0x00,0x00},
	{0x00,0x00,0x20,0x20,0x20,0x20,0x10,0x08,0x08,0xf8,0x00,0x00},
	{0x00,0x00,0x70,0x88,0x88,0x88,0x70,0x88,0x88,0x70,0x00,0x00},
	{0x00,0x00,0x70,0x08,0x08,0x78,0x88,0x88,0x88,0x70,0x00,0x00},
	{0x00,0x00,0x30,0x30,0x00,0x00,0x30,0x30,0x00,0x00,0x00,0x00},
	{0x40,0x20,0x60,0x60,0x00,0x00,0x60,0x60,0x00,0x00,0x00,0x00},
	{0x00,0x00,0x00,0x08,0x10,0x20,0x40,0x20,0x10,0x08,0x00,0x00},
	{0x00,0x00,0x00,0x00,0x00,0xf8,0x00,0xf8,0x00,0x00,0x00,0x00},
	{0x00,0x00,0x00,0x40,0x20,0x10,0x08,0x10,0x20,0x40,0x00,0x00},
	{0x00,0x00,0x20,0x00,0x20,0x10,0x08,0x08,0x88,0x70,0x00,0x00},
	{0x00,0x00,0x78,0x80,0x80,0xb8,0xa8,0xb8,0x88,0x70,0x00,0x00},
	{0x00,0x00,0x88,0x88,0xf8,0x88,0x88,0x50,0x50,0x20,0x00,0x00},
	{0x00,0x00,0xf0,0x88,0x88,0x88,0xf0,0x88,0x88,0xf0,0x00,0x00},
	{0x00,0x00,0x70,0x88,0x80,0x80,0x80,0x80,0x88,0x70,0x00,0x00},
	{0x00,0x00,0xf0,0x88,0x88,0x88,0x88,0x88,0x88,0xf0,0x00,0x00},
	{0x00,0x00,0xf8,0x80,0x80,0x80,0xf0,0x80,0x80,0xf8,0x00,0x00},
	{0x00,0x00,0x80,0x80,0x80,0x80,0xf0,0x80,0x80,0xf8,0x00,0x00},
	{0x00,0x00,0x70,0x88,0x88,0x98,0x80,0x80,0x88,0x70,0x00,0x00},
	{0x00,0x00,0x88,0x88,0x88,0x88,0xf8,0x88,0x88,0x88,0x00,0x00},
	{0x00,0x00,0xf8,0x20,0x20,0x20,0x20,0x20,0x20,0xf8,0x00,0x00},
	{0x00,0x00,0x70,0x88,0x88,0x08,0x08,0x08,0x08,0x08,0x00,0x00},
	{0x00,0x00,0x88,0x90,0xa0,0xc0,0xc0,0xa0,0x90,0x88,0x00,0x00},
	{0x00,0x00,0xf8,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x00,0x00},
	{0x00,0x00,0x88,0x88,0x88,0x88,0xa8,0xa8,0xd8,0x88,0x00,0x00},
	{0x00,0x00,0x88,0x88,0x88,0x88,0x98,0xa8,0xc8,0x88,0x00,0x00},
	{0x00,0x00,0x70,0x88,0x88,0x88,0x88,0x88,0x88,0x70,0x00,0x00},
	{0x00,0x00,0x80,0x80,0x80,0x80,0xf0,0x88,0x88,0xf0,0x00,0x00},
	{0x00,0x08,0x70,0xa8,0x88,0x88,0x88,0x88,0x88,0x70,0x00,0x00},
	{0x00,0x00,0x88,0x88,0x88,0x88,0xf0,0x88,0x88,0xf0,0x00,0x00},
	{0x00,0x00,0x70,0x88,0x08,0x08,0x70,0x80,0x88,0x70,0x00,0x00},
	{0x00,0x00,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0xf8,0x00,0x00},
	{0x00,0x00,0x70,0x88,0x88,0x88,0x88,0x88,0x88,0x88,0x00,0x00},
	{0x00,0x00,0x20,0x20,0x50,0x50,0x88,0x88,0x88,0x88,0x00,0x00},
	{0x00,0x00,0x88,0xd8,0xa8,0xa8,0x88,0x88,0x88,0x88,0x00,0x00},
	{0x00,0x00,0x88,0x88,0x50,0x20,0x50,0x88,0x88,0x88,0x00,0x00},
	{0x00,0x00,0x20,0x20,0x20,0x50,0x88,0x88,0x88,0x88,0x00,0x00},
	{0x00,0x00,0xf8,0x80,0x80,0x40,0x20,0x10,0x08,0xf8,0x00,0x00},
	{0x00,0x30,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x30,0x00},
	{0x04,0x04,0x08,0x08,0x10,0x10,0x20,0x20,0x40,0x40,0x00,0x00},
	{0x00,0x60,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x60,0x00},
	{0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x88,0x50,0x20,0x00,0x00},
	{0xfc,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
	{0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x20,0x40,0x00},
	{0x00,0x00,0x68,0x98,0x88,0x88,0x78,0x00,0x00,0x00,0x00,0x00},
	{0x00,0x00,0xf0,0x88,0x88,0x88,0xf0,0x80,0x80,0x80,0x00,0x00},
	{0x00,0x00,0x78,0x80,0x80,0x88,0x70,0x00,0x00,0x00,0x00,0x00},
	{0x00,0x00,0x78,0x88,0x88,0x88,0x78,0x08,0x08,0x08,0x00,0x00},
	{0x00,0x00,0x78,0x80,0xf8,0x88,0x70,0x00,0x00,0x00,0x00,0x00},
	{0x00,0x00,0x20,0x20,0x20,0x20,0x20,0x70,0x20,0x18,0x00,0x00},
	{0x70,0x08,0x78,0x88,0x88,0x88,0x78,0x00,0x00,0x00,0x00,0x00},
	{0x00,0x00,0x88,0x88,0x88,0x88,0xf0,0x80,0x80,0x80,0x00,0x00},
	{0x00,0x00,0x70,0x20,0x20,0x20,0x60,0x00,0x00,0x20,0x00,0x00},
	{0xc0,0x20,0x20,0x20,0x20,0x20,0x60,0x00,0x00,0x20,0x00,0x00},
	{0x00,0x00,0x88,0x90,0xe0,0xa0,0x90,0x80,0x80,0x80,0x00,0x00},
	{0x00,0x00,0x70,0x20,0x20,0x20,0x20,0x20,0x20,0x60,0x00,0x00},
	{0x00,0x00,0xa8,0xa8,0xa8,0xa8,0xf0,0x00,0x00,0x00,0x00,0x00},
	{0x00,0x00,0x88,0x88,0x88,0xc8,0xb0,0x00,0x00,0x00,0x00,0x00},
	{0x00,0x00,0x70,0x88,0x88,0x88,0x70,0x00,0x00,0x00,0x00,0x00},
	{0x80,0x80,0xf0,0x88,0x88,0x88,0xf0,0x00,0x00,0x00,0x00,0x00},
	{0x08,0x08,0x78,0x88,0x88,0x88,0x78,0x00,0x00,0x00,0x00,0x00},
	{0x00,0x00,0x80,0x80,0x80,0xc8,0xb0,0x00,0x00,0x00,0x00,0x00},
	{0x00,0x00,0xf0,0x08,0x70,0x80,0x78,0x00,0x00,0x00,0x00,0x00},
	{0x00,0x00,0x18,0x20,0x20,0x20,0x70,0x20,0x20,0x20,0x00,0x00},
	{0x00,0x00,0x68,0x98,0x88,0x88,0x88,0x00,0x00,0x00,0x00,0x00},
	{0x00,0x00,0x20,0x50,0x50,0x88,0x88,0x00,0x00,0x00,0x00,0x00},
	{0x00,0x00,0x50,0xa8,0xa8,0xa8,0xa8,0x00,0x00,0x00,0x00,0x00},
	{0x00,0x00,0x88,0x50,0x20,0x50,0x88,0x00,0x00,0x00,0x00,0x00},
	{0x70,0x08,0x78,0x88,0x88,0x88,0x88,0x00,0x00,0x00,0x00,0x00},
	{0x00,0x00,0xf8,0x40,0x20,0x10,0xf8,0x00,0x00,0x00,0x00,0x00},
	{0x10,0x20,0x20,0x20,0x20,0x40,0x20,0x20,0x20,0x20,0x10,0x00},
	{0x00,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x00},
	{0x40,0x20,0x20,0x20,0x20,0x10,0x20,0x20,0x20,0x20,0x40,0x00},
	{0x00,0x00,0x00,0x00,0x00,0x00,0xb0,0x68,0x00,0x00,0x00,0x00},
	{0x70,0xf8,0xa8,0xd8,0xf8,0xf8,0xa8,0xa8,0xa8,0xf8,0x70,0x00},
};

GLuint font_offset;

void init_font()
{
	int nchars;
	GLuint i;
	
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	font_offset = glGenLists (128);
	nchars = sizeof(font) / sizeof(font[0]);
	for (i = 0; i < nchars; ++i) {
		glNewList(font_offset + FONT_FIRST + i, GL_COMPILE);
		glBitmap(FONT_WIDTH, FONT_HEIGHT, 0, 0, FONT_WIDTH, 0.0, font[i]);
		glEndList();
	}
}

void draw_text(char *s, int x, int y) {
	glRasterPos2i(x, y);
	glPushAttrib (GL_LIST_BIT);
	glListBase(font_offset);
	glCallLists(strlen(s), GL_UNSIGNED_BYTE, (GLubyte *) s);
	glPopAttrib ();
}
