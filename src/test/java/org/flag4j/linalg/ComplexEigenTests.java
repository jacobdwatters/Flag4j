package org.flag4j.linalg;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.io.PrintOptions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ComplexEigenTests {
    static final long seed = 0xC0DE;
    CMatrix A;
    Complex128[][] entries;
    CVector exp;
    Complex128[] expEntries;
    CMatrix expV;
    Complex128[][] expVEntries;

    @Test
    void get2x2EigenValuesTests() {


        // --------------------- sub-case 1 ---------------------
        entries = new Complex128[][]{
                {new Complex128("0"), new Complex128("1")},
                {new Complex128("-1"), new Complex128("0")}};
        A = new CMatrix(entries);
        expEntries = new Complex128[]{new Complex128("0.0+1.0i"), new Complex128("0.0-1.0i")};
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.get2x2EigenValues(A));

        // --------------------- sub-case 2 ---------------------
        entries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0+1.0i")},
                {new Complex128("-0.0-1.0i"), new Complex128("0.0")}};
        A = new CMatrix(entries);
        expEntries = new Complex128[]{new Complex128("1"), new Complex128("-1")};
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.get2x2EigenValues(A));

        // --------------------- sub-case 3 ---------------------
        entries = new Complex128[][]{
                {new Complex128("2.344+3.56001i"), new Complex128("25.03915-8.255678i")},
                {new Complex128("-934.5"), new Complex128("0.0+26.0i")}};
        A = new CMatrix(entries);
        expEntries = new Complex128[]{new Complex128("25.91698338059983+170.13738197742643i"),
                new Complex128("-23.572983380599826-140.5773719774264i")};
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.get2x2EigenValues(A));

        // --------------------- sub-case 4 ---------------------
        entries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO}};
        A = new CMatrix(entries);
        expEntries = new Complex128[]{Complex128.ZERO, Complex128.ZERO};
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.get2x2EigenValues(A));
    }


    @Test
    void getEigenValuesTests() {
        PrintOptions.setPrecision(100);

        // ------------------- Sub-case 1 -------------------
        entries = new Complex128[][]{
                {new Complex128("0"), new Complex128("1")},
                {new Complex128("-1"), new Complex128("0")}};
        A = new CMatrix(entries);
        expEntries = new Complex128[]{new Complex128("0.0+1.0i"), new Complex128("0.0-1.0i")};
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.getEigenValues(A, seed));

        // ------------------- Sub-case 2 -------------------
        entries = new Complex128[][]{
                {new Complex128("0.30339+0.52411i"), new Complex128("0.31941+0.29588i"), new Complex128("0.56224+0.97309i")},
                {new Complex128("0.08611+0.04236i"), new Complex128("0.98099+0.54528i"), new Complex128("0.4917+0.78095i")},
                {new Complex128("0.45323+0.82093i"), new Complex128("0.28344+0.95413i"), new Complex128("0.31625+0.79709i")}};
        A = new CMatrix(entries);
        expEntries = new Complex128[]{new Complex128("1.2146241725922988 + 1.9378160847101924i"),
                new Complex128("-0.11326855369139792 - 0.42446262318817385i"),
                new Complex128("0.4992743810991004 + 0.3531265384779854i")};
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.getEigenValues(A, seed));

        // ------------------- Sub-case 3 -------------------
        entries = new Complex128[][]{
                {new Complex128("0.97857+0.91072i"), new Complex128("0.81174+0.08269i"), new Complex128("0.80316+0.49063i"), new Complex128("0.12095+0.27668i"), new Complex128("0.31818+0.38495i")},
                {new Complex128("0.67915+0.29834i"), new Complex128("0.2405+0.21984i"), new Complex128("0.9966+0.51097i"), new Complex128("0.9435+0.40336i"), new Complex128("0.32169+0.09356i")},
                {new Complex128("0.55321+0.53369i"), new Complex128("0.03852+0.57618i"), new Complex128("0.79603+0.68543i"), new Complex128("0.43236+0.23291i"), new Complex128("0.94369+0.08539i")},
                {new Complex128("0.31664+0.98003i"), new Complex128("0.40312+0.91131i"), new Complex128("0.97575+0.236i"), new Complex128("0.11995+0.50877i"), new Complex128("0.21747+0.83452i")},
                {new Complex128("0.54424+0.79567i"), new Complex128("0.2451+0.5495i"), new Complex128("0.78562+0.24197i"), new Complex128("0.71864+0.3659i"), new Complex128("0.28414+0.13826i")}};
        A = new CMatrix(entries);
        expEntries = new Complex128[]{
                new Complex128(2.7893912122770947, 2.2557188346117036),
                new Complex128(0.26992265297896606, -0.4597603145339981),
                new Complex128(-0.6030872920420072, -0.4435375080536591),
                new Complex128(-0.29662043545535155, 0.6229801159150405),
                new Complex128(0.25958386224131025, 0.48761887206092025)
        };
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.getEigenValues(A, seed, 200));

        // ------------------- Sub-case 4 -------------------
        double[][] entries1 = new double[][]{
                {0, 0, 0, 1},
                {0, 0, -1, 0},
                {0, 1, 0, 0},
                {-1, 0, 0, 0}};
        A = new CMatrix(entries1);
        expEntries = new Complex128[]{
                new Complex128(-2.7755575615628914E-17,-0.9999999999999998),
                new Complex128(-2.7755575615628914E-17, 0.9999999999999994),
                new Complex128(0, 1),
                new Complex128(0, -1)};
        exp = new CVector(expEntries);
        assertEquals(exp, Eigen.getEigenValues(A, seed, 40));

        // ------------------- Sub-case 5 -------------------
        entries = new Complex128[][]{
                {new Complex128("0.30339+0.52411i"), new Complex128("0.31941+0.29588i"), new Complex128("0.56224+0.97309i")},
                {new Complex128("0.08611+0.04236i"), new Complex128("0.98099+0.54528i"), new Complex128("0.4917+0.78095i")}};
        A = new CMatrix(entries);
        assertThrows(IllegalArgumentException.class, ()-> Eigen.getEigenValues(A));

        // ------------------- Sub-case 6 -------------------
        entries = new Complex128[][]{
                {new Complex128("0.30339+0.52411i"), new Complex128("0.31941+0.29588i")},
                {new Complex128("0.08611+0.04236i"), new Complex128("0.98099+0.54528i")},
                {new Complex128("0.45323+0.82093i"), new Complex128("0.28344+0.95413i")}};
        A = new CMatrix(entries);
        assertThrows(IllegalArgumentException.class, ()-> Eigen.getEigenValues(A));
    }


    @Test
    void getEigenVectorsTests() {
        // ------------------- Sub-case 1 -------------------
        entries = new Complex128[][]{
                {new Complex128(0), new Complex128(1)},
                {new Complex128(-1), new Complex128(0)}};
        A = new CMatrix(entries);
        expVEntries = new Complex128[][]{
                {new Complex128(0.0, 0.7071067811865475), new Complex128(0.7071067811865475)},
                {new Complex128(-0.7071067811865475), new Complex128(0.0, -0.7071067811865475)}
        };
        expV = new CMatrix(expVEntries);
        assertEquals(expV, Eigen.getEigenVectors(A, seed));

        // ------------------- Sub-case 2 -------------------
        entries = new Complex128[][]{
                {new Complex128("0.78906+0.50856i"), new Complex128("0.56982+0.15716i"), new Complex128("0.01619+0.49716i")},
                {new Complex128("0.64914+0.03802i"), new Complex128("0.41546+0.82783i"), new Complex128("0.10937+0.95944i")},
                {new Complex128("0.06341+0.45141i"), new Complex128("0.96949+0.33225i"), new Complex128("0.34728+0.84965i")}};
        A = new CMatrix(entries);
        expVEntries = new Complex128[][]{
                {new Complex128(0.39074716924687747, -0.268428344623814), new Complex128(0.2550510725241495, 0.07695739329205784), new Complex128(-0.7273193862283214, 0.25014930427788856)},
                {new Complex128(0.5819910600892099, -0.11004875164118963), new Complex128(-0.23846899693162998, -0.6471413682151337), new Complex128(-0.10097438362746816, 0.20534576443160502)},
                {new Complex128(0.5983605131211007, -0.2576882770014521), new Complex128(0.4990402817482533, 0.45202421866716674), new Complex128(0.017946251980771433, -0.5964453528250859)}
        };
        expV = new CMatrix(expVEntries);
        assertEquals(expV, Eigen.getEigenVectors(A, seed));
    }
}
