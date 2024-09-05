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

import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.TensorBase;
import org.flag4j.util.ErrorMessages;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.StringJoiner;

/**
 * The TensorWriter class provides several static methods for writing serialized
 * tensors, matrices, and vectors to a file.
 */
public final class TensorWriter {

    // TODO: Add options for matrix market format and csv format.

    private TensorWriter() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }

    /**
     * Writes a tensor to a file using a {@link TensorOutputStream}. If the write operation fails it will
     * terminate but no exception will be thrown. To determine if the write was successful, see the return value: if
     * the write was successful, true is returned; if the write fails,then false is returned.
     * is returned.
     * @param fileName File name, including extension, of the file to write the serialized matrix to.
     * @param src Source object to write to the specified file.
     * @return True if the write was successful. False if the write failed.
     */
    public static boolean write(String fileName, TensorBase src) {
        boolean successfulWrite = true;

        try (TensorOutputStream out = new TensorOutputStream(fileName)) {
            out.write(src);
        } catch (IOException e) {
            // Some exception was thrown.
            successfulWrite = false;
        }

        return successfulWrite;
    }


    /**
     * Writes the specified matrix to a csv file.
     * @param fileName File path to write matrix to.
     * @param src MatrixOld to write to csv file.
     * @param delimiter Delimiter to use in csv file.
     * @return True if the write was successful. False if the write failed.
     */
    public static boolean toCsv(String fileName, MatrixMixin src, String delimiter) {
        boolean successfulWrite = true;

        int numRows = src.numRows();
        int numCols = src.numCols();

        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(fileName))) {
            for (int i = 0; i < numRows; i++) {
                StringJoiner rowJoiner = new StringJoiner(delimiter);
                for (int j = 0; j < numCols; j++) {
                    rowJoiner.add(src.get(i, j).toString());
                }
                writer.write(rowJoiner.toString());
                writer.newLine();
            }
        } catch (IOException e) {
            successfulWrite = false;
        }

        return successfulWrite;
    }


    /**
     * Writes the specified matrix to a csv file.
     * @param fileName File path to write matrix to.
     * @param src MatrixOld to write to csv file.
     * @return True if the write was successful. False if the write failed.
     */
    public static boolean toCsv(String fileName, MatrixMixin src) {
        return toCsv(fileName, src, ", ");
    }
}
