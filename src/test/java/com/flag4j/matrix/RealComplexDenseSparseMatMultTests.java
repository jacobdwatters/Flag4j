package com.flag4j.matrix;

import com.flag4j.CMatrix;
import com.flag4j.Matrix;
import com.flag4j.Shape;
import com.flag4j.SparseCMatrix;
import com.flag4j.complex_numbers.CNumber;
import static com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseMatrixMultiplication.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class RealComplexDenseSparseMatMultTests {

    double[][] aEntries;
    CNumber[][] expCEntries;

    Matrix A;
    CMatrix expC;

    CNumber[] bEntries;
    int[] rowIndices, colIndices;
    SparseCMatrix B;
    Shape bShape;

    @Test
    void matMultTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i"), new CNumber("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new SparseCMatrix(bShape, bEntries, rowIndices, colIndices);
        expCEntries = new CNumber[][]{{new CNumber("-92.7375568794+927.8378999999999i"), new CNumber("0.00143541-0.000246i")},
                {new CNumber("-515.255376035+5155.1225i"), new CNumber("-10.7763114+1.84684i")},
                {new CNumber("-0.00012148943299999999+0.0012154999999999998i"), new CNumber("0.0")},
                {new CNumber("-11.4330901794+114.3879i"), new CNumber("-115804.09409999999+19846.46i")}};
        expC = new CMatrix(expCEntries);

        assertArrayEquals(expC.entries, standard(A.entries, A.shape, B.entries, B.rowIndices, B.colIndices, B.shape));
        assertArrayEquals(expC.entries, concurrentStandard(A.entries, A.shape, B.entries, B.rowIndices, B.colIndices, B.shape));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i"), new CNumber("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(3, 4);
        B = new SparseCMatrix(bShape, bEntries, rowIndices, colIndices);
        expCEntries = new CNumber[][]{{new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("-1.04985560794+10.503789999999999i"), new CNumber("-92.7375568794+927.8378999999999i"), new CNumber("-0.00011494769430000002+0.0011500500000000001i")},
                {new CNumber("-10881.6915+1864.9i"), new CNumber("6434.2545-1102.7i"), new CNumber("-10.7763114+1.84684i")}};
        expC = new CMatrix(expCEntries);

        assertArrayEquals(expC.entries, standard(B.entries, B.rowIndices, B.colIndices, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, concurrentStandard(B.entries, B.rowIndices, B.colIndices, B.shape, A.entries, A.shape));
    }


    @Test
    void matVecMultTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i"), new CNumber("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 0};
        bShape = new Shape(3, 1);
        B = new SparseCMatrix(bShape, bEntries, rowIndices, colIndices);
        expCEntries = new CNumber[][]{{new CNumber("-92.7361214694+927.8376539999999i")},
                {new CNumber("-526.0316874350001+5156.969340000001i")},
                {new CNumber("-0.00012148943299999999+0.0012154999999999998i")},
                {new CNumber("-115815.52719017939+19960.8479i")}};
        expC = new CMatrix(expCEntries);

        assertArrayEquals(expC.entries, standardVector(A.entries, A.shape, B.entries, B.rowIndices));
        assertArrayEquals(expC.entries, concurrentStandardVector(A.entries, A.shape, B.entries, B.rowIndices));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234},
                {-932.45},
                {123.445},
                {78.234}};
        A = new Matrix(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i"), new CNumber("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(3, 4);
        B = new SparseCMatrix(bShape, bEntries, rowIndices, colIndices);
        expCEntries = new CNumber[][]{{new CNumber("0.0")},
                {new CNumber("-1.04985560794+10.503789999999999i")},
                {new CNumber("-10881.6915+1864.9i")}};
        expC = new CMatrix(expCEntries);

        assertArrayEquals(expC.entries, standardVector(B.entries, B.rowIndices, B.colIndices, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, concurrentStandardVector(B.entries, B.rowIndices, B.colIndices, B.shape, A.entries, A.shape));
    }
}
