package com.flag4j.linalg.decompositions;

import com.flag4j.CMatrix;
import com.flag4j.Matrix;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class RealSchurTests {

    RealSchurDecomposition schur;
    double[][] aEntries, expTEntriesReal, expUEntriesReal;
    Matrix A;
    CNumber[][] expTEntries, expUEntries;
    CMatrix expT, expU;


    @Test
    void schurDecompSymmetricTest() {
//        // --------------------- Sub-case 1 ---------------------
//        aEntries = new double[][]{
//                {2, 1, 0, 0, 0, 0},
//                {1, 2, 1, 0, 0, 0},
//                {0, 1, 2, 1, 0, 0},
//                {0, 0, 1, 2, 1, 0},
//                {0, 0, 0, 1, 2, 1},
//                {0, 0, 0, 0, 1, 2}};
//        A = new Matrix(aEntries);
//
//        expTEntriesReal = new double[][]{
//                {3.801937735804834, 0, 0, 0, 0, 0},
//                {0, 3.246979603717471, 0, 0, 0, 0},
//                {0, 0, 2.4450418679126322, 0, 0, 0},
//                {0, 0, 0, 1.5549581320873678, 0, 0},
//                {0, 0, 0, 0, 0.7530203962825338, 0},
//                {0, 0, 0, 0, 0, 0.1980622641951618}};
//        expT = new CMatrix(expTEntriesReal);
//
//        expUEntriesReal = new double[][]{
//                {-0.23192061392493687, -0.41790650594093925, 0.5211208891696028, -0.5211208891696018, 0.4179065059412745, -0.2319206139243295},
//                {-0.4179065059420311, -0.5211208891689969, 0.2319206139243297, 0.23192061392433058, -0.5211208891696022, 0.417906505941275},
//                {-0.5211208891699384, -0.23192061392357372, -0.4179065059412759, 0.4179065059412738, 0.23192061392432975, -0.5211208891696023},
//                {-0.5211208891692659, 0.23192061392508628, -0.41790650594127504, -0.41790650594127443, 0.23192061392433003, 0.5211208891696025},
//                {-0.41790650594051937, 0.5211208891702087, 0.23192061392433008, -0.23192061392432967, -0.521120889169603, -0.41790650594127454},
//                {-0.23192061392372396, 0.41790650594161244, 0.5211208891696025, 0.5211208891696018, 0.4179065059412755, 0.23192061392432967}};
//        expU = new CMatrix(expUEntriesReal);
//
//        schur = new RealSchurDecomposition().decompose(A);
//
//        assertEquals(expT, schur.getT().roundToZero());
//        assertEquals(expU, schur.getU());
//
//        schur = new RealSchurDecomposition(false).decompose(A);
//        assertEquals(expT, schur.getT().roundToZero());
//        assertNull(schur.getU());
    }
}
