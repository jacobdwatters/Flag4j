package org.flag4j.operations_old.dense_sparse.real_complex;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CooCMatrix;
import org.flag4j.arrays_old.sparse.CooMatrix;
import org.flag4j.arrays_old.sparse.CooVector;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.flag4j.operations_old.dense_sparse.coo.real_complex.RealComplexDenseSparseMatrixMultiplication.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class RealComplexDenseSparseMatMultTests {

    double[][] aEntries;
    CNumber[][] expCEntries, bComplexEntries;

    MatrixOld A;
    CMatrixOld expC, BComplex;

    CNumber[] bEntries;
    CooCMatrix BSparseComplex;
    Shape shape;

    CooMatrix ASparse;
    CooVector ACooVector;
    double[] aSparseEntries;

    int[] rowIndices, colIndices;

    @Test
    void matMultTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new MatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i"), new CNumber("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        shape = new Shape(3, 2);
        BSparseComplex = new CooCMatrix(shape, bEntries, rowIndices, colIndices);
        expCEntries = new CNumber[][]{{new CNumber("-92.7375568794+927.8378999999999i"), new CNumber("0.00143541-0.000246i")},
                {new CNumber("-515.255376035+5155.1225i"), new CNumber("-10.7763114+1.84684i")},
                {new CNumber("-0.00012148943299999999+0.0012154999999999998i"), new CNumber("0.0")},
                {new CNumber("-11.4330901794+114.3879i"), new CNumber("-115804.09409999999+19846.46i")}};
        expC = new CMatrixOld(expCEntries);

        assertArrayEquals(expC.entries, standard(A.entries, A.shape, BSparseComplex.entries, BSparseComplex.rowIndices, BSparseComplex.colIndices, BSparseComplex.shape));
        assertArrayEquals(expC.entries, concurrentStandard(A.entries, A.shape, BSparseComplex.entries, BSparseComplex.rowIndices, BSparseComplex.colIndices, BSparseComplex.shape));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new MatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i"), new CNumber("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        shape = new Shape(3, 4);
        BSparseComplex = new CooCMatrix(shape, bEntries, rowIndices, colIndices);
        expCEntries = new CNumber[][]{{new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("-1.04985560794+10.503789999999999i"), new CNumber("-92.7375568794+927.8378999999999i"), new CNumber("-0.00011494769430000002+0.0011500500000000001i")},
                {new CNumber("-10881.6915+1864.9i"), new CNumber("6434.2545-1102.7i"), new CNumber("-10.7763114+1.84684i")}};
        expC = new CMatrixOld(expCEntries);

        assertArrayEquals(expC.entries, standard(BSparseComplex.entries, BSparseComplex.rowIndices, BSparseComplex.colIndices, BSparseComplex.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, concurrentStandard(BSparseComplex.entries, BSparseComplex.rowIndices, BSparseComplex.colIndices, BSparseComplex.shape, A.entries, A.shape));

        // ---------------------- Sub-case 3 ----------------------
        aSparseEntries = new double[]{1, 9.43};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{2, 1};
        shape = new Shape(4, 3);
        ASparse = new CooMatrix(shape, aSparseEntries, rowIndices, colIndices);
        bComplexEntries = new CNumber[][]{
                {new CNumber(1.34, 13.4), new CNumber(234.6, 6)},
                {new CNumber(-9.55, 1.9414), new CNumber(9, 1)},
                {new CNumber(0.9923, -985.2), new CNumber(9234)}};
        BComplex = new CMatrixOld(bComplexEntries);
        expCEntries = new CNumber[][]{{new CNumber("0.9923-985.2i"), new CNumber("9234.0")},
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("-90.0565+18.307402i"), new CNumber("84.87+9.43i")},
                {new CNumber("0.0"), new CNumber("0.0")}};
        expC = new CMatrixOld(expCEntries);

        assertArrayEquals(expC.entries, standard(ASparse.entries, ASparse.rowIndices, ASparse.colIndices, ASparse.shape,
                BComplex.entries, BComplex.shape));
        assertArrayEquals(expC.entries, concurrentStandard(ASparse.entries, ASparse.rowIndices, ASparse.colIndices, ASparse.shape,
                BComplex.entries, BComplex.shape));

        // ---------------------- Sub-case 4 ----------------------
        aSparseEntries = new double[]{1, 9.43};
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{2, 1};
        shape = new Shape(2, 3);
        ASparse = new CooMatrix(shape, aSparseEntries, rowIndices, colIndices);
        bComplexEntries = new CNumber[][]{
                {new CNumber(1.34, 13.4), new CNumber(234.6, 6)},
                {new CNumber(-9.55, 1.9414), new CNumber(9, 1)},
                {new CNumber(0.9923, -985.2), new CNumber(9234)}};
        BComplex = new CMatrixOld(bComplexEntries);
        expCEntries = new CNumber[][]{{new CNumber("0.0"), new CNumber("2212.278+56.58i"), new CNumber("1.34+13.4i")},
                {new CNumber("0.0"), new CNumber("84.87+9.43i"), new CNumber("-9.55+1.9414i")},
                {new CNumber("0.0"), new CNumber("87076.62"), new CNumber("0.9923-985.2i")}};
        expC = new CMatrixOld(expCEntries);

        assertArrayEquals(expC.entries, standard(BComplex.entries, BComplex.shape,
                ASparse.entries, ASparse.rowIndices, ASparse.colIndices, ASparse.shape));
        assertArrayEquals(expC.entries, concurrentStandard(BComplex.entries, BComplex.shape,
                ASparse.entries, ASparse.rowIndices, ASparse.colIndices, ASparse.shape));
    }


    @Test
    void matVecMultTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new MatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i"), new CNumber("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 0};
        shape = new Shape(3, 1);
        BSparseComplex = new CooCMatrix(shape, bEntries, rowIndices, colIndices);
        expCEntries = new CNumber[][]{{new CNumber("-92.7361214694+927.8376539999999i")},
                {new CNumber("-526.0316874350001+5156.969340000001i")},
                {new CNumber("-0.00012148943299999999+0.0012154999999999998i")},
                {new CNumber("-115815.52719017939+19960.8479i")}};
        expC = new CMatrixOld(expCEntries);

        assertArrayEquals(expC.entries, standardVector(A.entries, A.shape, BSparseComplex.entries, BSparseComplex.rowIndices));
        assertArrayEquals(expC.entries, concurrentStandardVector(A.entries, A.shape, BSparseComplex.entries, BSparseComplex.rowIndices));
        assertArrayEquals(expC.entries, blockedVector(A.entries, A.shape, BSparseComplex.entries, BSparseComplex.rowIndices));
        assertArrayEquals(expC.entries, concurrentBlockedVector(A.entries, A.shape, BSparseComplex.entries, BSparseComplex.rowIndices));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234},
                {-932.45},
                {123.445},
                {78.234}};
        A = new MatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i"), new CNumber("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        shape = new Shape(3, 4);
        BSparseComplex = new CooCMatrix(shape, bEntries, rowIndices, colIndices);
        expCEntries = new CNumber[][]{{new CNumber("0.0")},
                {new CNumber("-1.04985560794+10.503789999999999i")},
                {new CNumber("-10881.6915+1864.9i")}};
        expC = new CMatrixOld(expCEntries);

        assertArrayEquals(expC.entries, standardVector(BSparseComplex.entries, BSparseComplex.rowIndices, BSparseComplex.colIndices, BSparseComplex.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, concurrentStandardVector(BSparseComplex.entries, BSparseComplex.rowIndices, BSparseComplex.colIndices, BSparseComplex.shape, A.entries, A.shape));

        // ---------------------- Sub-case 3 ----------------------
        aSparseEntries = new double[]{1, 9.43};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{2, 1};
        shape = new Shape(4, 3);
        ASparse = new CooMatrix(shape, aSparseEntries, rowIndices, colIndices);
        bComplexEntries = new CNumber[][]{
                {new CNumber(1.34, 13.4)},
                {new CNumber(-9.55, 1.9414)},
                {new CNumber(0.9923, -985.2)}};
        BComplex = new CMatrixOld(bComplexEntries);
        expCEntries = new CNumber[][]{{new CNumber("0.9923-985.2i")},
                {new CNumber("0.0")},
                {new CNumber("-90.0565+18.307402i")},
                {new CNumber("0.0")}};
        expC = new CMatrixOld(expCEntries);

        assertArrayEquals(expC.entries, standardVector(ASparse.entries, ASparse.rowIndices, ASparse.colIndices, ASparse.shape,
                BComplex.entries, BComplex.shape));
        assertArrayEquals(expC.entries, concurrentStandardVector(ASparse.entries, ASparse.rowIndices, ASparse.colIndices, ASparse.shape,
                BComplex.entries, BComplex.shape));

        // ---------------------- Sub-case 4 ----------------------
        aSparseEntries = new double[]{9.43};
        rowIndices = new int[]{1};
        ACooVector = new CooVector(2, aSparseEntries, rowIndices);
        bComplexEntries = new CNumber[][]{
                {new CNumber(1.34, 13.4), new CNumber(234.6, 6)},
                {new CNumber(-9.55, 1.9414), new CNumber(9, 1)},
                {new CNumber(0.9923, -985.2), new CNumber(9234)}};
        BComplex = new CMatrixOld(bComplexEntries);
        expCEntries = new CNumber[][]{{new CNumber("2212.278+56.58i")},
                {new CNumber("84.87+9.43i")},
                {new CNumber("87076.62")}};
        expC = new CMatrixOld(expCEntries);

        assertArrayEquals(expC.entries, standardVector(BComplex.entries, BComplex.shape,
                ACooVector.entries, ACooVector.indices));
        assertArrayEquals(expC.entries, concurrentStandardVector(BComplex.entries, BComplex.shape,
                ACooVector.entries, ACooVector.indices));
        assertArrayEquals(expC.entries, blockedVector(BComplex.entries, BComplex.shape,
                ACooVector.entries, ACooVector.indices));
        assertArrayEquals(expC.entries, concurrentBlockedVector(BComplex.entries, BComplex.shape,
                ACooVector.entries, ACooVector.indices));

    }
}
