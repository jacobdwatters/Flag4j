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

package org.flag4j.operations.dense.real_complex;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.dense.CMatrix;
import org.flag4j.dense.Matrix;
import org.junit.jupiter.api.Test;

import static org.flag4j.operations.dense.real_complex.RealComplexDenseMatrixMultTranspose.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class RealComplexDenseMatMultTransposeTests {
    double[][] aEntries;

    Matrix A;
    CMatrix expC;

    @Test
    void matMultTestCase() {
        CNumber[][] bEntries;
        CMatrix B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber("1.666+1.0i"), new CNumber("11.5-9.123i")},
                {new CNumber("-0.0-0.9345341i"), new CNumber("88.234")},
                {new CNumber("0.0"), new CNumber("0.00002+85.23i")}};
        B = new CMatrix(bEntries).T();
        expC = A.mult(B.T());

        assertArrayEquals(expC.entries, multTranspose(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, multTransposeBlocked(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, multTransposeConcurrent(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, multTransposeBlockedConcurrent(A.entries, A.shape, B.entries, B.shape));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342}};
        A = new Matrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber("1.666+1.0i"), new CNumber("11.5-9.123i")},
                {new CNumber("-0.0-0.9345341i"), new CNumber("88.234")},
                {new CNumber("0.0"), new CNumber("0.00002+85.23i")}};
        B = new CMatrix(bEntries).T();
        expC = A.mult(B.T());

        assertArrayEquals(expC.entries, multTranspose(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, multTransposeBlocked(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, multTransposeConcurrent(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, multTransposeBlockedConcurrent(A.entries, A.shape, B.entries, B.shape));

        // ---------------------- Sub-case 3 ----------------------
        aEntries = new double[][]{
                {1.1234, 99.234},
                {-932.45, 551.35},
                {0.000123, -0.92342}};
        A = new Matrix(aEntries);
        bEntries = new CNumber[][]{
                {new CNumber("1.666+1.0i"), new CNumber("11.5-9.123i"), new CNumber("0.0")},
                {new CNumber("-0.0-0.9345341i"), new CNumber("88.234"), new CNumber("0.00002+85.23i")}};
        B = new CMatrix(bEntries).T();
        expC = B.mult(A.T());

        assertArrayEquals(expC.entries, multTranspose(B.entries, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, multTransposeBlocked(B.entries, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, multTransposeConcurrent(B.entries, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, multTransposeBlockedConcurrent(B.entries, B.shape, A.entries, A.shape));
    }
}
