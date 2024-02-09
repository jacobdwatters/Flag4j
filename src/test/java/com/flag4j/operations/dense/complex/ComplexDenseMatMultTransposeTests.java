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

package com.flag4j.operations.dense.complex;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.dense.CMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ComplexDenseMatMultTransposeTests {
    CNumber[][] entriesA, entriesB;
    CMatrix A, B;
    CNumber[] exp, act;

    @Test
    void squareTestCase() {
        entriesA = new CNumber[][]{{new CNumber("1.34+14.3i"), new CNumber("1.51-9.51i"), new CNumber("71.5i")},
                {new CNumber("13.55+0i"), new CNumber("-0.00014+14.661i"), new CNumber("7.398+0.98134i")},
                {new CNumber("0.0014+9.55i"), new CNumber("-45.6i"), new CNumber("-94.51+0i")}};
        entriesB = new CNumber[][]{{new CNumber("0"), new CNumber("-94.1-65.1123i"), new CNumber("1.44")},
                {new CNumber("-0.000013+1i"), new CNumber("-1i"), new CNumber("80.441-9.331i")},
                {new CNumber("-8.314-1i"), new CNumber("814.4i"), new CNumber("1.4556+9.4414i")}};
        A = new CMatrix(entriesA);
        B = new CMatrix(entriesB);
        exp = A.mult(B.T()).entries;

        // ------------ Sub-case 1 ------------
        act = ComplexDenseMatrixMultTranspose.multTranspose(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 2 ------------
        act = ComplexDenseMatrixMultTranspose.multTransposeBlocked(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 3 ------------
        act = ComplexDenseMatrixMultTranspose.multTransposeConcurrent(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 4 ------------
        act = ComplexDenseMatrixMultTranspose.multTransposeBlockedConcurrent(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);
    }


    @Test
    void rectangleTestCase() {
        entriesA = new CNumber[][]{{new CNumber("1.34+14.3i"), new CNumber("1.51-9.51i"), new CNumber("71.5i")},
                {new CNumber("13.55+0i"), new CNumber("-0.00014+14.661i"), new CNumber("7.398+0.98134i")},
                {new CNumber("0.0014+9.55i"), new CNumber("-45.6i"), new CNumber("-94.51+0i")}};
        entriesB = new CNumber[][]{{new CNumber("0"), new CNumber("-94.1-65.1123i")},
                {new CNumber("-0.000013+1i"), new CNumber("-1i")},
                {new CNumber("-8.314-1i"), new CNumber("814.4i")}};
        exp = new CNumber[]{new CNumber("81.00998037 - 592.9408763700001i"), new CNumber("-57434.09811-1434.3904819999998i"),
                new CNumber("-75.18663199817999 - 15.557191352999999i"), new CNumber("-2059.597296+5142.659675i"),
                new CNumber("831.3561400000001+94.51059280000001i"), new CNumber("576.090725-77867.69015722i")};
        A = new CMatrix(entriesA);
        B = new CMatrix(entriesB).T();
        exp = A.mult(B.T()).entries;

        // ------------ Sub-case 1 ------------
        act = ComplexDenseMatrixMultTranspose.multTranspose(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 2 ------------
        act = ComplexDenseMatrixMultTranspose.multTransposeBlocked(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 3 ------------
        act = ComplexDenseMatrixMultTranspose.multTransposeConcurrent(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 4 ------------
        act = ComplexDenseMatrixMultTranspose.multTransposeBlockedConcurrent(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);
    }
}
