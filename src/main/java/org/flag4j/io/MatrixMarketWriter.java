/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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


import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;


/**
 * <p></p>A utility class for writing matrices (from the Flag4j library) to a file in the
 * <a href="https://math.nist.gov/MatrixMarket/formats.html">Matrix Market Exchange Format</a>.
 *
 * <p>This class supports writing both dense and sparse matrices (real or complex) to the
 * Matrix Market format. Dense matrices are written in the <em>array</em> format, while
 * sparse (both COO and CSR) matrices are written in the <em>coordinate</em> (COO) format.
 *
 * <p>All matrices are currently written in general form and the symmetry is not considered.
 *
 * <p><h2>Currently Supported Matrix Types</h2>
 * <ul>
 *   <li>{@code Matrix} (real dense)</li>
 *   <li>{@code CMatrix} (complex dense)</li>
 *   <li>{@code CooMatrix} (real sparse COO)</li>
 *   <li>{@code CsrMatrix} (real sparse CSR, converted to COO)</li>
 *   <li>{@code CooCMatrix} (complex sparse COO)</li>
 *   <li>{@code CsrCMatrix} (complex sparse CSR, converted to COO)</li>
 * </ul>
 *
 * <p>Attempting to write a matrix type not listed above will result in an
 * {@link IllegalArgumentException}.
 *
 * @see MatrixMarketReader
 * @see MatrixMarketHeader
 */
public final class MatrixMarketWriter {

    private MatrixMarketWriter() {
        // Hide default constructor for utility class.
    }


    /**
     * <p>Writes a matrix to a file in Matrix Market Exchange Format file.
     * <p>If {@code mat} is a CSR matrix, it will be converted to a COO matrix and saved as a coordinate matrix.
     * <p>Currently supported matrix types:
     * <ul>
     *   <li>{@code Matrix} (real dense)</li>
     *   <li>{@code CMatrix} (complex dense)</li>
     *   <li>{@code CooMatrix} (real sparse COO)</li>
     *   <li>{@code CsrMatrix} (real sparse CSR, converted to COO)</li>
     *   <li>{@code CooCMatrix} (complex sparse COO)</li>
     *   <li>{@code CsrCMatrix} (complex sparse CSR, converted to COO)</li>
     * </ul>
     *
     * @param filePath Path of the file to write to.
     * @param mat Matrix to write to file.
     * @param comments Comments to prepend to file. Each comment will be written to its own line.
     * May be {@code null}; in this case the parameter will be ignored.
     * @throws IOException If an I/O error occurs.
     * @throws IllegalArgumentException If {@code mat} is not a supported matrix type for writing to a
     * Matrix Market Exchange Format file.
     */
    public static void write(String filePath, MatrixMixin<?, ?, ?, ?> mat, String... comments) throws IOException {
        boolean isDense = (mat instanceof Matrix || mat instanceof CMatrix);
        boolean isComplex = (mat instanceof CMatrix || mat instanceof CooCMatrix || mat instanceof CsrCMatrix);

        int[] shapeData = isDense
                ? new int[]{mat.numRows(), mat.numCols()}
                : new int[]{mat.numRows(), mat.numCols(), mat.dataLength()};

        MatrixMarketHeader header = new MatrixMarketHeader(
                MatrixMarketHeader.MMObject.MATRIX,
                isDense ? MatrixMarketHeader.MMFormat.ARRAY : MatrixMarketHeader.MMFormat.COORDINATE,
                isComplex ? MatrixMarketHeader.MMField.COMPLEX : MatrixMarketHeader.MMField.REAL,
                MatrixMarketHeader.MMSymmetry.GENERAL,
                comments
        );

        // Transpose to ensure colum-major ordering as required by Matrix Market Exchange Format.
        mat = mat.T();

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            writeHeader(writer, header, shapeData);

            if (isDense) writeDense(writer, mat);
            else writeSparse(writer, mat);
        }
    }


    /**
     * Writes the header of the Matrix Market Exchange Format to the file.
     * @param writer The {@link BufferedWriter} to use when writing to file.
     * @param header The Matrix Market Exchange Format header to write to the file.
     * @param shapeData Contains, in order, either the number of rows and number of columns, <i>or</i> the number of rows, columns,
     * and the number of non-zero data.
     * @throws IOException If an I/O error occurs.
     */
    private static void writeHeader(
            BufferedWriter writer, MatrixMarketHeader header, int... shapeData) throws IOException {
        writer.write(header.toString());
        writer.newLine();
        for(int value : shapeData)
            writer.write(value + " ");
        writer.newLine();
    }


    /**
     * Writes the data of a dense matrix to a Matrix Market Exchange Format file.
     * @param writer The writer to be used when writing to file.
     * @param mat The matrix to write.
     * @throws IOException If an I/O error occurs.
     * @throws IllegalArgumentException If {@code mat} is not a valid type for which writing to Matrix Market Exchange Format is
     * supported.
     */
    private static void writeDense(BufferedWriter writer, MatrixMixin<?, ?, ?, ?> mat) throws IOException {
        if(mat instanceof Matrix) {
            writeDenseArray(writer, ((Matrix) mat).data);
        } else if(mat instanceof CMatrix) {
            writeDenseArray(writer, ((CMatrix) mat).data);
        } else {
            throw new IllegalArgumentException(
                    "Matrix type not supported for writing to Matrix Market Exchange Format: "
                            + mat.getClass().getName());
        }
    }


    /**
     * Writes the data of a sparse matrix to a Matrix Market Exchange Format file.
     * @param writer The writer to be used when writing to file.
     * @param mat The matrix to write.
     * @throws IOException If an I/O error occurs.
     * @throws IllegalArgumentException If {@code mat} is not a valid type for which writing to Matrix Market Exchange Format is
     * supported.
     */
    private static void writeSparse(BufferedWriter writer, MatrixMixin<?, ?, ?, ?> mat) throws IOException {
        if(mat instanceof CooMatrix) {
            CooMatrix matrix = (CooMatrix) mat;
            writeCooArray(writer, matrix.data, matrix.rowIndices, matrix.colIndices);
        } else if(mat instanceof CsrMatrix) {
            CooMatrix matrix = ((CsrMatrix) mat).toCoo();
            writeCooArray(writer, matrix.data, matrix.rowIndices, matrix.colIndices);
        } else if(mat instanceof CooCMatrix) {
            CooCMatrix matrix = (CooCMatrix) mat;
            writeCooArray(writer, matrix.data, matrix.rowIndices, matrix.colIndices);
        } else if(mat instanceof CsrCMatrix) {
            CooCMatrix matrix = ((CsrCMatrix) mat).toCoo();
            writeCooArray(writer, matrix.data, matrix.rowIndices, matrix.colIndices);
        } else {
            throw new IllegalArgumentException(
                    "Matrix type not supported for writing to Matrix Market Exchange Format: "
                            + mat.getClass().getName());
        }
    }


    /**
     * Writes a dense array to a file such that each entry is on its own line.
     * @param writer The {@link BufferedWriter} to use when writing to file.
     * @param arr The array to write to the file.
     * @throws IOException If an I/O error occurs.
     */
    private static void writeDenseArray(BufferedWriter writer, double[] arr) throws IOException {
        for (double value : arr) {
            writer.write(String.valueOf(value));
            writer.newLine();
        }
    }


    /**
     * Writes a dense array to a file such that each entry is on its own line.
     * @param writer The {@link BufferedWriter} to use when writing to file.
     * @param arr The array to write to the file.
     * @throws IOException If an I/O error occurs.
     */
    private static void writeDenseArray(BufferedWriter writer, Complex128[] arr) throws IOException {
        for (Complex128 value : arr) {
            writer.write(value.re + " " + value.im);
            writer.newLine();
        }
    }


    /**
     * Writes a sparse COO array to a file such that each entry and its indices are on its own line according to the Matrix Market
     * Exchange Format.
     * @param writer The {@link BufferedWriter} to use when writing to file.
     * @param data The non-zero data to write to the file.
     * @param rowIndices The non-zero row indices to write to the file.
     * @param colIndices The non-zero column indices to write to the file.
     * @throws IOException If an I/O error occurs.
     */
    private static void writeCooArray(
            BufferedWriter writer, double[] data, int[] rowIndices, int[] colIndices) throws IOException {
        for (int i=0, size=data.length; i<size; i++) {
            // Ensure indices are 1-based as specified in Matrix Market Exchange format.
            writer.write((rowIndices[i] + 1) + " " + (colIndices[i] + 1) + " " + data[i]);
            writer.newLine();
        }
    }


    /**
     * Writes a sparse COO array to a file such that each entry and its indices are on its own line according to the Matrix Market
     * Exchange Format.
     * @param writer The {@link BufferedWriter} to use when writing to file.
     * @param data The non-zero data to write to the file.
     * @param rowIndices The non-zero row indices to write to the file.
     * @param colIndices The non-zero column indices to write to the file.
     * @throws IOException If an I/O error occurs.
     */
    private static void writeCooArray
    (BufferedWriter writer, Complex128[] data, int[] rowIndices, int[] colIndices) throws IOException {
        for (int i=0, size=data.length; i<size; i++) {
            // Ensure indices are 1-based as specified in Matrix Market Exchange format.
            writer.write((rowIndices[i] + 1) + " " + (colIndices[i] + 1) + " " + data[i].re + " " + data[i].im);
            writer.newLine();
        }
    }
}
