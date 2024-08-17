/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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


import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.TensorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.util.ErrorMessages;

import java.io.IOException;

/**
 * The TensorReader class provides several static methods for reading serialized
 * tensors, matrices, and vectors from a file.
 */
public class TensorReader {

    /*
        TODO:
            - Add options for matrix market format and csv format.
             TensorOld reader should be instantiable object where the constructor takes an argument for the format,
             i.e. serialized, matrix market, or csv.
            - Make the type to write a generic so only one non-static method is needed for read.
     */

    private TensorReader() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Reads a serialized {@link TensorOld tensor} from a specified file using a {@link TensorInputStream}. If an exception is
     * thrown while reading, the result will be null.
     * @param fileName File name, including extension, of the file containing the serialized tensor.
     * @return The deserialized tensor from the specified file. If an exception is thrown while reading,
     * the result will be null.
     */
    public static TensorOld readTensor(String fileName) {
        TensorOld tensor;

        try (TensorInputStream in = new TensorInputStream(fileName)) {
            tensor = in.readTensor();
        } catch (IOException | ClassNotFoundException e) {
            // An Exception was thrown.
            tensor = null;
        }

        return tensor;
    }


    /**
     * Reads a serialized {@link MatrixOld matrix} from a specified file using a {@link TensorInputStream}. If an exception is thrown while reading,
     * the result will be null.
     * @param fileName File name, including extension, of the file containing the serialized matrix.
     * @return The deserialized matrix from the specified file. If an exception is thrown while reading,
     * the result will be null.
     */
    public static MatrixOld readMatrix(String fileName) {
        MatrixOld matrix;

        try (TensorInputStream in = new TensorInputStream(fileName)) {
            matrix = in.readMatrix();
        } catch (IOException | ClassNotFoundException e) {
            // An Exception was thrown.
            matrix = null;
        }

        return matrix;
    }


    /**
     * Reads a serialized {@link VectorOld vector} from a specified file using a {@link TensorInputStream}. If an exception is
     * thrown while reading, the result will be null.
     * @param fileName File name, including extension, of the file containing the serialized vector.
     * @return The deserialized vector from the specified file. If an exception is thrown while reading,
     * the result will be null.
     */
    public static VectorOld readVector(String fileName) {
        VectorOld vector;

        try (TensorInputStream in = new TensorInputStream(fileName)) {
            vector = in.readVector();
        } catch (IOException | ClassNotFoundException e) {
            // An Exception was thrown.
            vector = null;
        }

        return vector;
    }
}
