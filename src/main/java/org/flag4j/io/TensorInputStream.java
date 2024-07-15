/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
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

package org.flag4j.io;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.dense.Vector;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;


/**
 * A {@code TensorInputStream} obtains bytes from a system file using a {@link FileInputStream} containing a serialized
 * tensor, matrix, or vector. Then, deserializes that object using an {@link ObjectInputStream}.
 */
public class TensorInputStream extends InputStream {

    /**
     * Stream for loading bytes from a system file.
     */
    private final FileInputStream fileIn;
    /**
     * Stream for deserializing an object.
     */
    private final ObjectInputStream objectIn;


    /**
     * Creates {@link TensorInputStream} which will open a connection with a file through an {@link ObjectInputStream}
     * and a {@link FileInputStream}. This {@link TensorInputStream} can be used to deserialize a tensor,
     * matrix, or vector which is stored in a system file.
     * @param fileName File name, including extension, of tensor, matrix, or vector to read.
     * @throws IOException If an I/O error occurs while reading the stream header.
     */
    public TensorInputStream(String fileName) throws IOException {
        fileIn = new FileInputStream(fileName);
        objectIn = new ObjectInputStream(fileIn);
    }


    /**
     * Reads a serialized {@link Tensor tesnor} from a file.
     * @return The deserialized {@link Tensor tensor object} stored in the specified file.
     */
    public Tensor readTensor() throws IOException, ClassNotFoundException {
        // Deserialize tensor.
        return (Tensor) objectIn.readObject();
    }


    /**
     * Reads a serialized {@link Matrix matrix} from a file.
     * @return The deserialized {@link Matrix matrix object} stored in the specified file.
     */
    public Matrix readMatrix() throws IOException, ClassNotFoundException {
        // Deserialize matrix.
        return (Matrix) objectIn.readObject();
    }


    /**
     * Reads a serialized {@link Vector vector} from a file.
     * @return The deserialized {@link Vector vector object} stored in the specified file.
     */
    public Vector readVector() throws IOException, ClassNotFoundException {
        // Deserialize vector.
        return (Vector) objectIn.readObject();
    }


    /**
     * Reads a byte of data. This method will block if no input is available. The read operation is forwarded
     * to {@link ObjectInputStream#read()}.
     *
     * @return the next byte of data, or {@code -1} if the end of the
     * stream is reached.
     * @throws IOException If an I/O error occurs.
     */
    @Override
    public int read() throws IOException {
        return objectIn.read();
    }


    /**
     * Closes the stream. This method must be called to release any resources associated with the stream.
     * @throws IOException If an I/O exception has occurred.
     */
    @Override
    public void close() throws IOException {
        objectIn.close();
        fileIn.close();
    }
}
