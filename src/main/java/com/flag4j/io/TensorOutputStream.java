/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.flag4j.io;

import com.flag4j.core.TensorBase;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;


/**
 * A class for writing tensors to a file in various formats.
 */
public class TensorOutputStream extends OutputStream {

    /**
     * Stream for writing data to a file.
     */
    private final FileOutputStream fileOut;
    /**
     * Stream for serializing object.
     */
    private final ObjectOutputStream objectOut;


    /**
     * Creates {@link TensorOutputStream} which will use an {@link ObjectOutputStream} to write serialized data to a
     * {@link FileOutputStream} which will in-turn write the serialized data to a file.
     * @param fileName File name, including extension, of file to write data to.
     * @throws IOException If an I/O error occurs while reading the stream header.
     */
    public TensorOutputStream(String fileName) throws IOException {
        fileOut = new FileOutputStream(fileName);
        objectOut = new ObjectOutputStream(fileOut);
    }


    /**
     * Writes a {@link TensorBase} object to a file by serializing the object.
     * @param A Object to write to file. This may be a real/complex dense or sparse tensor, matrix, or vector.
     */
    public void write(TensorBase<?, ?, ?, ?, ?, ?> A) throws IOException {
        objectOut.writeObject(A);
    }


    /**
     * Writes a bite to the output stream. This method will forward the write to {@link ObjectOutputStream#write(int)}.
     *
     * @param b the {@code byte}.
     * @throws IOException if an I/O error occurs. In particular,
     *                     an {@code IOException} may be thrown if the
     *                     output stream has been closed.
     */
    @Override
    public void write(int b) throws IOException {
        objectOut.write(b);
    }


    /**
     * Closes the stream. This method must be called to release any resources associated with the stream.
     * @throws IOException If an I/O exception has occurred.
     */
    @Override
    public void close() throws IOException {
        objectOut.close();
        fileOut.close();
    }
}
