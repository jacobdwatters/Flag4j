package org.flag4j.linalg;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

class RealEigenTests {
    static final long seed = 0xC0DE;
    Matrix A;
    double[][] entries;
    CVector exp;
    Complex128[] expEntries;
    CMatrix expV;
    Complex128[][] expVEntries;

    @Test
    void get2x2EigenValuesTests() {
        // ------------------- Sub-case 1 -------------------
        entries = new double[][]{
                {1, 2}, {3, 4}};
        A = new Matrix(entries);
        expEntries = new Complex128[]{new Complex128("-0.3722813232690143"), new Complex128("5.372281323269013")};
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.get2x2EigenValues(A));

        // ------------------- Sub-case 2 -------------------
        entries = new double[][]{
                {10.5, 2.4},
                {-0.0024, 215.66}};
        A = new Matrix(entries);
        expEntries = new Complex128[]{new Complex128("10.500028075652104"), new Complex128("215.65997192434787")};
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.get2x2EigenValues(A));

        // ------------------- Sub-case 3 -------------------
        entries = new double[][]{
                {0, 1},
                {-1, 0}};
        A = new Matrix(entries);
        expEntries = new Complex128[]{new Complex128("0.9999999999999998i"), new Complex128("-0.9999999999999998i")};
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.get2x2EigenValues(A));

        // ------------------- Sub-case 4 -------------------
        entries = new double[][]{
                {0, 0},
                {0, 0}};
        A = new Matrix(entries);
        expEntries = new Complex128[]{Complex128.ZERO, Complex128.ZERO};
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.get2x2EigenValues(A));
    }


    @Test
    void getEigenValuesTests() {
        // ------------------- Sub-case 1 -------------------
        entries = new double[][]{
                {1, 2}, {3, 4}};
        A = new Matrix(entries);
        expEntries = new Complex128[]{new Complex128("-0.3722813232690143"), new Complex128("5.372281323269013")};
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.getEigenValues(A, seed));

        // ------------------- Sub-case 2 -------------------
        entries = new double[][]{
                {0, 1},
                {-1, 0}};
        A = new Matrix(entries);
        expEntries = new Complex128[]{new Complex128("0.9999999999999998i"), new Complex128("-0.9999999999999998i")};
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.getEigenValues(A, seed));

        // ------------------- Sub-case 3 -------------------
        entries = new double[][]{
                {0.4864, 0.85113, 0.96095},
                {0.87509, 0.41948, 0.73852},
                {0.47208, 0.79501, 0.41394}};
        A = new Matrix(entries);
        expEntries = new Complex128[]{new Complex128("1.9923303019116854"),
                new Complex128("-0.33625515095584274+0.14219266015547333i"),
                new Complex128("-0.33625515095584274-0.14219266015547333i")};
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.getEigenValues(A, seed));

        // ------------------- Sub-case 4 -------------------
        entries = new double[][]{
                {1, 0, 0},
                {0, 2, 0},
                {0, 0, 3}};
        A = new Matrix(entries);
        expEntries = new Complex128[]{new Complex128(1),
                new Complex128(2),
                new Complex128(3)};
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.getEigenValues(A, seed));

        // ------------------- Sub-case 5 -------------------
        entries = new double[][]{
                {2, -3, 0},
                {3,  2, 0},
                {0,  0, 4}};
        A = new Matrix(entries);
        expEntries = new Complex128[]{
                new Complex128("1.999999999999999 + 2.999999999999998i"),
                new Complex128("1.999999999999999 - 2.999999999999998i"),
                new Complex128(4)};
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.getEigenValues(A, seed));

        // ------------------- Sub-case 6 -------------------
        entries = new double[][]{
                {0.75988, 0.86932, 0.95331, 0.60455, 0.46308, 0.58296, 0.76133, 0.19523, 0.65878, 0.76903, 0.65807, 0.95184},
                {0.03246, 0.11275, 0.2476, 0.61116, 0.5048, 0.2674, 0.22478, 0.90103, 0.78827, 0.81668, 0.13316, 0.81351},
                {0.18713, 0.61423, 0.17935, 0.84529, 0.9906, 0.85287, 0.8771, 0.33781, 0.5458, 0.75962, 0.40587, 0.74471},
                {0.07549, 0.51289, 0.0771, 0.87689, 0.09414, 0.87488, 0.88761, 0.67612, 0.28392, 0.49383, 0.81843, 0.95646},
                {0.41044, 0.66063, 0.36204, 0.6973, 0.83859, 0.26924, 0.82409, 0.27248, 0.00113, 0.47992, 0.25885, 0.66848},
                {0.77308, 0.08378, 0.14225, 0.71248, 0.79402, 0.76337, 0.08424, 0.78964, 0.95543, 0.91629, 0.43461, 0.32733},
                {0.3913, 0.914, 0.67269, 0.12027, 0.71771, 0.28293, 0.03802, 0.84081, 0.72483, 0.04088, 0.51326, 0.15226},
                {0.30991, 0.56374, 0.09985, 0.01886, 0.64519, 0.76862, 0.5787, 0.08207, 0.61055, 0.7182, 0.99097, 0.00778},
                {0.50291, 0.45905, 0.14384, 0.83067, 0.47036, 0.82197, 0.19375, 0.87983, 0.84801, 0.95461, 0.10998, 0.66883},
                {0.04512, 0.12456, 0.47719, 0.66192, 0.18928, 0.38026, 0.5962, 0.93584, 0.23512, 0.27215, 0.71346, 0.2241},
                {0.98206, 0.79166, 0.34167, 0.83044, 0.94746, 0.0393, 0.16674, 0.05952, 0.33419, 0.32641, 0.35915, 0.03885},
                {0.32974, 0.25636, 0.11247, 0.0328, 0.62604, 0.6062, 0.42941, 0.63618, 0.39747, 0.08202, 0.16033, 0.26547}};
        A = new Matrix(entries);
        expEntries = new Complex128[]{
                new Complex128(-1.2286261410556234), new Complex128(5.918529433580702),
                new Complex128("-0.3714369468278178 + 1.0811961967602406i"), new Complex128("-0.3714369468278178 - 1.0811961967602406i"),
                new Complex128(0.7658108389129894), new Complex128("0.22177258075901635 + 0.6185897919333341i"),
                new Complex128("0.22177258075901635 - 0.6185897919333341i"), new Complex128("0.2910077285360231 + 0.45339806202918165i"),
                new Complex128("0.2910077285360231 - 0.45339806202918165i"), new Complex128("-0.3413607889178575 + 0.17801686614592319i"),
                new Complex128("-0.3413607889178575 - 0.17801686614592319i"), new Complex128(0.34002072146321344)};
        exp = new CVector(expEntries);

        assertEquals(exp, Eigen.getEigenValues(A, seed));

        // ------------------- Sub-case 7 -------------------
        entries = new double[][]{
                {0, 0, 0, 1},
                {0, 0, -1, 0},
                {0, 1, 0, 0},
                {-1, 0, 0, 0}};
        A = new Matrix(entries);
        expEntries = new Complex128[]{
                new Complex128(5.5511151231257815E-17,0.9999999999999982),
                new Complex128(5.5511151231257815E-17, -0.9999999999999982),
                new Complex128(0, 0.9999999999999998),
                new Complex128(0, -0.9999999999999998)};
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.getEigenValues(A, seed, 40));

        // ------------------- Sub-case 8 -------------------
        entries = new double[][]{
                {0, 0, 0, 1},
                {0, 0, -1, 0},
                {0, 1, 0, 0}};
        A = new Matrix(entries);
        assertThrows(IllegalArgumentException.class, ()-> Eigen.getEigenValues(A));

        // ------------------- Sub-case 9 -------------------
        entries = new double[][]{
                {0, 0},
                {0, 0},
                {0, 1}};
        A = new Matrix(entries);
        assertThrows(IllegalArgumentException.class, ()-> Eigen.getEigenValues(A));
    }


    @Test
    void getEigenVectorsTest() {
        // ------------------- Sub-case 1 -------------------
        entries = new double[][]{
                {1, 2}, {3, 4}};
        A = new Matrix(entries);
        expVEntries = new Complex128[][]{
                {new Complex128("0.41597355791928425"), new Complex128("-0.8245648401323938")},
                {new Complex128("0.9093767091321241"), new Complex128("0.5657674649689923")}};
        expV = new CMatrix(expVEntries);

        assertEquals(expV, Eigen.getEigenVectors(A, seed));

        // ------------------- Sub-case 2 -------------------
        entries = new double[][]{
                {0, 1},
                {-1, 0}};
        A = new Matrix(entries);
        expVEntries = new Complex128[][]{
                {new Complex128(0.0, 0.7071067811865475), new Complex128(0.7071067811865475)},
                {new Complex128(-0.7071067811865475), new Complex128(0.0, -0.7071067811865475)}
        };
        expV = new CMatrix(expVEntries);
        assertEquals(expV, Eigen.getEigenVectors(A, seed));

        // ------------------- Sub-case 3 -------------------
        entries = new double[][]{
                {0.4864, 0.85113, 0.96095},
                {0.87509, 0.41948, 0.73852},
                {0.47208, 0.79501, 0.41394}};
        A = new Matrix(entries);
        expVEntries = new Complex128[][]{
                {new Complex128(0.6443341561451386, 0.0), new Complex128(-0.3314770517125463, -0.24793593934966027), new Complex128(0.24793593934966018, 0.3314770517125463)},
                {new Complex128(0.5880515587233901, 0.0), new Complex128(-0.27126365550635384, 0.5694436225180889), new Complex128(-0.5694436225180888, 0.2712636555063539)},
                {new Complex128(0.4889057777401735, 0.0), new Complex128(0.5607228366620323, -0.3411607019972403), new Complex128(0.34116070199724025, -0.5607228366620324)}
        };
        expV = new CMatrix(expVEntries);

        assertEquals(expV, Eigen.getEigenVectors(A, seed));

        // ------------------- Sub-case 4 -------------------
        entries = new double[][]{
                {0, 0, 0, 1},
                {0, 0, -1, 0},
                {0, 1, 0, 0},
                {-1, 0, 0, 0}};
        A = new Matrix(entries);
        expVEntries = new Complex128[][]{
                {new Complex128(0.707106781186547, 7.850462293418876E-17), new Complex128(7.850462293418876E-17, 0.707106781186547), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0)},
                {new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(-0.7071067811865475, 0.0), new Complex128(0.0, -0.7071067811865475)},
                {new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.7071067811865475), new Complex128(0.7071067811865475, 0.0)},
                {new Complex128(1.962615573354719E-16, 0.7071067811865469), new Complex128(0.7071067811865469, 1.962615573354719E-16), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0)}
        };
        expV = new CMatrix(expVEntries);

        assertEquals(expV, Eigen.getEigenVectors(A, seed, 40));
    }
}
