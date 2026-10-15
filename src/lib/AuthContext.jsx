import React, { createContext, useContext, useState, useEffect } from "react";
import { base44 } from "@/api/base44Client";

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [isLoadingAuth, setIsLoadingAuth] = useState(true);
  const [isLoadingPublicSettings, setIsLoadingPublicSettings] = useState(false);
  const [authError, setAuthError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    
    // Check initial user authentication state
    const checkAuth = async () => {
      try {
        if (base44?.auth?.getUser) {
          const currentUser = await base44.auth.getUser();
          if (isMounted) setUser(currentUser);
        }
      } catch (err) {
        if (isMounted) setAuthError({ type: "auth_required", message: err.message });
      } finally {
        if (isMounted) setIsLoadingAuth(false);
      }
    };

    checkAuth();
    return () => { isMounted = false; };
  }, []);

  const navigateToLogin = () => {
    window.location.href = "/login";
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoadingAuth,
        isLoadingPublicSettings,
        authError,
        navigateToLogin,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    return {
      user: null,
      isLoadingAuth: false,
      isLoadingPublicSettings: false,
      authError: null,
      navigateToLogin: () => { window.location.href = "/login"; },
    };
  }
  return context;
};
