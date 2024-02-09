package com.flag4j.operations.dense.complex;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.dense.CMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ComplexMatrixMultiplyTests {
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
        exp = new CNumber[]{new CNumber("81.00998037 - 592.9408763700001i"), new CNumber("-57434.09811-1434.3904819999998i"), new CNumber("-640.4024000000001 - 654.41632i"),
                new CNumber("-75.18663199817999 - 15.557191352999999i"), new CNumber("-2059.597296+5142.659675i"), new CNumber("157.80583458399997+1250.622723044i"),
                new CNumber("831.3561400000001+94.51059280000001i"), new CNumber("576.090725-77867.69015722i"), new CNumber("-563.06034-4546.664314000001i")};

        A = new CMatrix(entriesA);
        B = new CMatrix(entriesB);

        // ------------ Sub-case 1 ------------
        act = ComplexDenseMatrixMultiplication.standard(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 2 ------------
        act = ComplexDenseMatrixMultiplication.reordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 3 ------------
        act = ComplexDenseMatrixMultiplication.blocked(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 4 ------------
        act = ComplexDenseMatrixMultiplication.blockedReordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 5 ------------
        act = ComplexDenseMatrixMultiplication.concurrentStandard(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 6 ------------
        act = ComplexDenseMatrixMultiplication.concurrentReordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 7 ------------
        act = ComplexDenseMatrixMultiplication.concurrentBlocked(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 8 ------------
        act = ComplexDenseMatrixMultiplication.concurrentBlockedReordered(A.entries, A.shape, B.entries, B.shape);
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
        B = new CMatrix(entriesB);

        // ------------ Sub-case 1 ------------
        act = ComplexDenseMatrixMultiplication.standard(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 2 ------------
        act = ComplexDenseMatrixMultiplication.reordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 3 ------------
        act = ComplexDenseMatrixMultiplication.blocked(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 4 ------------
        act = ComplexDenseMatrixMultiplication.blockedReordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 5 ------------
        act = ComplexDenseMatrixMultiplication.concurrentStandard(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 6 ------------
        act = ComplexDenseMatrixMultiplication.concurrentReordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 7 ------------
        act = ComplexDenseMatrixMultiplication.concurrentBlocked(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 8 ------------
        act = ComplexDenseMatrixMultiplication.concurrentBlockedReordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);
    }


    @Test
    void columnVectorTestCase() {
        entriesA = new CNumber[][]{{new CNumber("1.34+14.3i"), new CNumber("1.51-9.51i"), new CNumber("71.5i")},
                {new CNumber("13.55+0i"), new CNumber("-0.00014+14.661i"), new CNumber("7.398+0.98134i")},
                {new CNumber("0.0014+9.55i"), new CNumber("-45.6i"), new CNumber("-94.51+0i")}};
        entriesB = new CNumber[][]{{new CNumber("0")},
                {new CNumber("-0.000013+1i")},
                {new CNumber("-8.314-1i")}};
        exp = new CNumber[]{new CNumber("81.00998037-592.9408763700001i"),
                new CNumber("-75.18663199817999-15.557191352999999i"),
                new CNumber("831.3561400000001+94.51059280000001i")};

        A = new CMatrix(entriesA);
        B = new CMatrix(entriesB);

        // ------------ Sub-case 1 ------------
        act = ComplexDenseMatrixMultiplication.standardVector(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 2 ------------
        act = ComplexDenseMatrixMultiplication.blockedVector(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 3 ------------
        act = ComplexDenseMatrixMultiplication.concurrentStandardVector(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 4 ------------
        act = ComplexDenseMatrixMultiplication.concurrentBlockedVector(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);
    }
}
