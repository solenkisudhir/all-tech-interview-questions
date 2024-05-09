# Types of Encoding Techniques
_The process of conversion of data from one form to another form is known as Encoding._ It is used to transform the data so that data can be supported and used by different systems. Encoding works similarly to converting temperature from centigrade to Fahrenheit, as it just gets converted in another form, but the original value always remains the same. Encoding is used in mainly two fields:

*   **Encoding in Electronics:** In electronics, encoding refers to converting analog signals to digital signals.
*   **Encoding in Computing:** In computing, encoding is a process of converting data to an equivalent cipher by applying specific code, letters, and numbers to the data.

#### Note: Encoding is different from encryption as its main purpose is not to hide the data but to convert it into a format so that it can be properly consumed.

In this topic, we are going to discuss the different types of encoding techniques that are used in computing.

Type of Encoding Technique
--------------------------

![Types of Encoding Techniques](https://static.javatpoint.com/tutorial/machine-learning/images/types-of-encoding-techniques.png)

*   **Character Encoding**
*   **Image & Audio and Video Encoding**

### Character Encoding

**_Character encoding encodes characters into bytes_**. It informs the computers how to interpret the zero and ones into real characters, numbers, and symbols. The computer understands only binary data; hence it is required to convert these characters into numeric codes. To achieve this, each character is converted into binary code, and for this, text documents are saved with encoding types. It can be done by pairing numbers with characters. If we don't apply character encoding, our website will not display the characters and text in a proper format. Hence it will decrease the readability, and the machine would not be able to process data correctly. Further, character encoding makes sure that each character has a proper representation in computer or binary format.

There are different types of Character Encoding techniques, which are given below:

1.  **HTML Encoding**
2.  **URL Encoding**
3.  **Unicode Encoding**
4.  **Base64 Encoding**
5.  **Hex Encoding**
6.  **ASCII Encoding**

### HTML Encoding

HTML encoding is used to display an HTML page in a proper format. With encoding, a web browser gets to know that which character set to be used.

In HTML, there are various characters used in HTML Markup such as <, >. To encode these characters as content, we need to use an encoding.

### URL Encoding

URL (Uniform resource locator) Encoding is used to **_convert characters in such a format that they can be transmitted over the internet_**. It is also known as percent-encoding. The URL Encoding is performed to send the URL to the internet using the ASCII character-set. Non-ASCII characters are replaced with a %, followed by the hexadecimal digits.

### UNICODE Encoding

Unicode is an encoding standard for a universal character set. It allows encoding, represent, and handle the text represented in most of the languages or writing systems that are available worldwide. It provides a code point or number for each character in every supported language. It can represent approximately all the possible characters possible in all the languages. A particular sequence of bits is known as a coding unit.

A UNICODE standard can use 8, 16, or 32 bits to represent the characters.

The Unicode standard defines Unicode Transformation Format (UTF) to encode the code points.

UNICODE Encoding standard has the following UTF schemes:

*   **UTF-8 Encoding**  
    The UTF8 is defined by the UNICODE standard, which is variable-width character encoding used in Electronics Communication. UTF-8 is capable of encoding all 1,112,064 valid character code points in Unicode using one to four one-byte (8-bit) code units.
*   **UTF-16 Encoding**  
    UTF16 Encoding represents a character's code points using one of two 16-bits integers.
*   **UTF-32 Encoding**  
    UTF32 Encoding represents each code point as 32-bit integers.

### Base64 Encoding

Base64 Encoding is used to encode binary data into equivalent ASCII Characters. The Base64 encoding is used in the Mail system as mail systems such as SMTP can't work with binary data because they accept ASCII textual data only. It is also used in simple HTTP authentication to encode the credentials. Moreover, it is also used to transfer the binary data into cookies and other parameters to make data unreadable to prevent tampering. If an image or another file is transferred without Base64 encoding, it will get corrupted as the mail system is not able to deal with binary data.

Base64 represents the data into blocks of 3 bytes, where each byte contains 8 bits; hence it represents 24 bits. These 24 bits are divided into four groups of 6 bits. Each of these groups or chunks are converted into equivalent Base64 value.

### ASCII Encoding

**American Standard Code for Information Interchange** (ASCII) is a type of character-encoding. It was the first character encoding standard released in the year 1963.

Th ASCII code is used to represent English characters as numbers, where each letter is assigned with a number from **0 to 127.** Most modern character-encoding schemes are based on ASCII, though they support many additional characters. It is a single byte encoding only using the bottom 7 bits. In an ASCII file, each alphabetic, numeric, or special character is represented with a 7-bit binary number. Each character of the keyboard has an equivalent ASCII value.

Image and Audio & Video Encoding
--------------------------------

Image and audio & video encoding are performed to save storage space. A media file such as image, audio, and video are encoded to save them in a more efficient and compressed format.

These encoded files contain the same content with usually similar quality, but in compressed size, so that they can be saved within less space, can be transferred easily via mail, or can be downloaded on the system.

We can understand it as a . WAV audio file is converted into .MP3 file to reduce the size by 1/10th to its original size.

* * *