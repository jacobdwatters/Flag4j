package org.flag4j.linalg.operations.dense.complex;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.linalg.operations.dense.field_ops.DenseFieldMatrixMultiplication;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ComplexMatrixMultiplyTests {
    Complex128[][] entriesA, entriesB;
    CMatrix A, B;
    Field<Complex128>[] exp, act;

    @Test
    void squareTestCase() {
        entriesA = new Complex128[][]{{new Complex128("1.34+14.3i"), new Complex128("1.51-9.51i"), new Complex128("71.5i")},
                {new Complex128("13.55+0i"), new Complex128("-0.00014+14.661i"), new Complex128("7.398+0.98134i")},
                {new Complex128("0.0014+9.55i"), new Complex128("-45.6i"), new Complex128("-94.51+0i")}};
        entriesB = new Complex128[][]{{new Complex128("0"), new Complex128("-94.1-65.1123i"), new Complex128("1.44")},
                {new Complex128("-0.000013+1i"), new Complex128("-1i"), new Complex128("80.441-9.331i")},
                {new Complex128("-8.314-1i"), new Complex128("814.4i"), new Complex128("1.4556+9.4414i")}};
        exp = new Complex128[]{new Complex128("81.00998037 - 592.9408763700001i"), new Complex128("-57434.09811-1434.3904819999998i"), new Complex128("-640.4024000000001 - 654.41632i"),
                new Complex128("-75.18663199817999 - 15.557191352999999i"), new Complex128("-2059.597296+5142.659675i"), new Complex128("157.80583458399997+1250.622723044i"),
                new Complex128("831.3561400000001+94.51059280000001i"), new Complex128("576.090725-77867.69015722i"), new Complex128("-563.06034-4546.664314000001i")};

        A = new CMatrix(entriesA);
        B = new CMatrix(entriesB);

        // ------------ Sub-case 1 ------------
        act = DenseFieldMatrixMultiplication.standard(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 2 ------------
        act = DenseFieldMatrixMultiplication.reordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 3 ------------
        act = DenseFieldMatrixMultiplication.blocked(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 4 ------------
        act = DenseFieldMatrixMultiplication.blockedReordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 5 ------------
        act = DenseFieldMatrixMultiplication.concurrentStandard(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 6 ------------
        act = DenseFieldMatrixMultiplication.concurrentReordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 7 ------------
        act = DenseFieldMatrixMultiplication.concurrentBlocked(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 8 ------------
        act = DenseFieldMatrixMultiplication.concurrentBlockedReordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);
    }


    @Test
    void rectangleTestCase() {
        entriesA = new Complex128[][]{{new Complex128("1.34+14.3i"), new Complex128("1.51-9.51i"), new Complex128("71.5i")},
                {new Complex128("13.55+0i"), new Complex128("-0.00014+14.661i"), new Complex128("7.398+0.98134i")},
                {new Complex128("0.0014+9.55i"), new Complex128("-45.6i"), new Complex128("-94.51+0i")}};
        entriesB = new Complex128[][]{{new Complex128("0"), new Complex128("-94.1-65.1123i")},
                {new Complex128("-0.000013+1i"), new Complex128("-1i")},
                {new Complex128("-8.314-1i"), new Complex128("814.4i")}};
        exp = new Complex128[]{new Complex128("81.00998037 - 592.9408763700001i"), new Complex128("-57434.09811-1434.3904819999998i"),
                new Complex128("-75.18663199817999 - 15.557191352999999i"), new Complex128("-2059.597296+5142.659675i"),
                new Complex128("831.3561400000001+94.51059280000001i"), new Complex128("576.090725-77867.69015722i")};

        A = new CMatrix(entriesA);
        B = new CMatrix(entriesB);

        // ------------ Sub-case 1 ------------
        act = DenseFieldMatrixMultiplication.standard(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 2 ------------
        act = DenseFieldMatrixMultiplication.reordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 3 ------------
        act = DenseFieldMatrixMultiplication.blocked(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 4 ------------
        act = DenseFieldMatrixMultiplication.blockedReordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 5 ------------
        act = DenseFieldMatrixMultiplication.concurrentStandard(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 6 ------------
        act = DenseFieldMatrixMultiplication.concurrentReordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 7 ------------
        act = DenseFieldMatrixMultiplication.concurrentBlocked(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 8 ------------
        act = DenseFieldMatrixMultiplication.concurrentBlockedReordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);
    }


    @Test
    void columnVectorTestCase() {
        entriesA = new Complex128[][]{{new Complex128("1.34+14.3i"), new Complex128("1.51-9.51i"), new Complex128("71.5i")},
                {new Complex128("13.55+0i"), new Complex128("-0.00014+14.661i"), new Complex128("7.398+0.98134i")},
                {new Complex128("0.0014+9.55i"), new Complex128("-45.6i"), new Complex128("-94.51+0i")}};
        entriesB = new Complex128[][]{{new Complex128("0")},
                {new Complex128("-0.000013+1i")},
                {new Complex128("-8.314-1i")}};
        exp = new Complex128[]{new Complex128("81.00998037-592.9408763700001i"),
                new Complex128("-75.18663199817999-15.557191352999999i"),
                new Complex128("831.3561400000001+94.51059280000001i")};

        A = new CMatrix(entriesA);
        B = new CMatrix(entriesB);

        // ------------ Sub-case 1 ------------
        act = DenseFieldMatrixMultiplication.standardVector(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 2 ------------
        act = DenseFieldMatrixMultiplication.blockedVector(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 3 ------------
        act = DenseFieldMatrixMultiplication.concurrentStandardVector(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 4 ------------
        act = DenseFieldMatrixMultiplication.concurrentBlockedVector(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);
    }
}
