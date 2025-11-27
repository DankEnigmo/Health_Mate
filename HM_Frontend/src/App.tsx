import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import FrontPage from "./pages/FrontPage"; // New FrontPage
import LoginPage from "./pages/LoginPage"; // Unified LoginPage
import RoleSelectionPage from "./pages/RoleSelectionPage"; // Modified RoleSelectionPage
import CaregiverDashboard from "./pages/caregiver/CaregiverDashboard";
import CaregiverCalendar from "./pages/caregiver/CaregiverCalendar";
import CaregiverAlerts from "./pages/caregiver/CaregiverAlerts";
import CaregiverFallDetection from "./pages/caregiver/CaregiverFallDetection";
import CaregiverSettings from "./pages/caregiver/CaregiverSettings";
import CaregiverMedications from "./pages/caregiver/CaregiverMedications";
import NotFound from "./pages/NotFound";
import MainLayout from "./components/MainLayout";
import PatientLayout from "./components/patient/PatientLayout";
import PatientDashboard from "./pages/patient/PatientDashboard";
import PatientCalendar from "./pages/patient/PatientCalendar";
import PatientSettings from "./pages/patient/PatientSettings";
import PatientMessageInput from "./pages/patient/PatientMessageInput";
import PatientMedications from "./pages/patient/PatientMedications";
import { SessionContextProvider, useSupabase } from "./components/SessionContextProvider";
import React from "react";

const queryClient = new QueryClient();

const ProtectedRoute: React.FC<{ children: React.ReactNode; allowedRoles?: ('patient' | 'caregiver')[] }> = ({ children, allowedRoles }) => {
  const { session, profile, loading } = useSupabase();

  if (loading) {
    return <div className="flex items-center justify-center min-h-screen">Loading authentication...</div>;
  }

  if (!session) {
    return <Navigate to="/" replace />; // Redirect to FrontPage if not authenticated
  }

  if (!profile?.role) {
    return <Navigate to="/select-role" replace />; // Redirect to role selection if no role is set
  }

  if (allowedRoles && !allowedRoles.includes(profile.role)) {
    // If role is set but not allowed for this route, redirect to their dashboard
    if (profile.role === 'patient') {
      return <Navigate to="/patient/message-input" replace />;
    } else if (profile.role === 'caregiver') {
      return <Navigate to="/caregiver/dashboard" replace />;
    }
    return <Navigate to="/" replace />; // Fallback to front page
  }

  return <>{children}</>;
};

const AppContent = () => {
  return (
    <Routes>
      <Route path="/" element={<FrontPage />} /> {/* New Front Page */}
      <Route path="/auth" element={<LoginPage />} /> {/* Unified Login/Signup */}
      <Route path="/select-role" element={<RoleSelectionPage />} /> {/* Role selection after login */}

      {/* Caregiver Routes - Protected and role-specific */}
      <Route path="/caregiver/dashboard" element={<ProtectedRoute allowedRoles={['caregiver']}><MainLayout><CaregiverDashboard /></MainLayout></ProtectedRoute>} />
      <Route path="/caregiver/calendar" element={<ProtectedRoute allowedRoles={['caregiver']}><MainLayout><CaregiverCalendar /></MainLayout></ProtectedRoute>} />
      <Route path="/caregiver/alerts" element={<ProtectedRoute allowedRoles={['caregiver']}><MainLayout><CaregiverAlerts /></MainLayout></ProtectedRoute>} />
      <Route path="/caregiver/fall-detection" element={<ProtectedRoute allowedRoles={['caregiver']}><MainLayout><CaregiverFallDetection /></MainLayout></ProtectedRoute>} />
      <Route path="/caregiver/medications" element={<ProtectedRoute allowedRoles={['caregiver']}><MainLayout><CaregiverMedications /></MainLayout></ProtectedRoute>} />
      <Route path="/caregiver/settings" element={<ProtectedRoute allowedRoles={['caregiver']}><MainLayout><CaregiverSettings /></MainLayout></ProtectedRoute>} />

      {/* Patient Routes - Protected and role-specific */}
      <Route path="/patient" element={<Navigate to="/patient/message-input" replace />} />
      <Route path="/patient/message-input" element={<ProtectedRoute allowedRoles={['patient']}><PatientLayout><PatientMessageInput /></PatientLayout></ProtectedRoute>} />
      <Route path="/patient/dashboard" element={<ProtectedRoute allowedRoles={['patient']}><PatientLayout><PatientDashboard /></PatientLayout></ProtectedRoute>} />
      <Route path="/patient/calendar" element={<ProtectedRoute allowedRoles={['patient']}><PatientLayout><PatientCalendar /></PatientLayout></ProtectedRoute>} />
      <Route path="/patient/medications" element={<ProtectedRoute allowedRoles={['patient']}><PatientLayout><PatientMedications /></PatientLayout></ProtectedRoute>} />
      <Route path="/patient/settings" element={<ProtectedRoute allowedRoles={['patient']}><PatientLayout><PatientSettings /></PatientLayout></ProtectedRoute>} />

      {/* Catch-all for 404 */}
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <SessionContextProvider>
          <AppContent />
        </SessionContextProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;