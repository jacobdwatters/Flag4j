/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
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


import com.flag4j.Matrix;
import com.flag4j.util.ErrorMessages;

import java.io.IOException;

/**
 * The TensorReader class provides several static methods for reading serialized
 * tensors, matrices, and vectors from a file.
 */
public class TensorReader {

    private TensorReader() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Reads a serialized matrix from a specified file using a {@link TensorInputStream}. If an exception is thrown while reading,
     * the result will be null.
     * @param fileName File name, including extension, of the file containing the serialized matrix.
     * @return The deserialized matrix from the specified file. If an exception is thrown while reading,
     * the result will be null.
     */
    public static Matrix readMatrix(String fileName) {
        Matrix A;

        try {
            TensorInputStream in = new TensorInputStream(fileName);
            A = in.readMatrix();
            in.close();

        } catch (IOException | ClassNotFoundException e) {
            A = null;
        }

        return A;
    }
}
