import cv2

def draw_multiline_text_with_border(image, text, position, font, font_scale, font_color, thickness, border_color, border_thickness, max_width):
    words = text.split(' ')
    space_width, _ = cv2.getTextSize(' ', font, font_scale, thickness)[0]
    max_height = 0
    lines = []
    current_line = ''
    current_width = 0

    for word in words:
        word_width, word_height = cv2.getTextSize(word, font, font_scale, thickness)[0]
        if current_width + word_width > max_width:
            lines.append(current_line)
            current_line = word
            current_width = word_width + space_width
        else:
            current_line += ' ' + word if current_line else word
            current_width += word_width + space_width
        max_height = max(max_height, word_height)

    if current_line:
        lines.append(current_line)

    y = position[1]
    for line in lines:
        text_width, text_height = cv2.getTextSize(line, font, font_scale, thickness)[0]
        x = (max_width - text_width) // 2  # Center alignment

        # 先绘制黑色边框文字
        cv2.putText(image, line, (x, y), font, font_scale, border_color, border_thickness)
        # 再绘制白色文字
        cv2.putText(image, line, (x, y), font, font_scale, font_color, thickness)

        y += text_height + 5  # Adding line spacing
