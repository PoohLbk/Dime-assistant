import React from 'react';
import { Link } from 'react-router-dom';
import { UserX, LogIn } from 'lucide-react';

export default function UserNotRegisteredError() {
  return (
    <div className="min-h-screen bg-[#07080A] text-[#F2F5F7] flex flex-col items-center justify-center p-6 text-center font-body">
      <div className="w-16 h-16 rounded-2xl bg-[#FF5252]/10 border border-[#FF5252]/30 flex items-center justify-center text-[#FF5252] mb-6 shadow-[0_0_30px_rgba(255,82,82,0.2)]">
        <UserX className="w-8 h-8" />
      </div>
      <h1 className="text-2xl font-black tracking-tight mb-2 uppercase">Account Not Found</h1>
      <p className="text-xs text-[#7A838D] max-w-sm mb-6 leading-relaxed">
        Your user session is recognized, but no matching registered account was found in the system database.
      </p>
      <Link
        to="/register"
        className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-[#2ED88A] text-[#07080A] font-bold text-xs uppercase tracking-wider hover:bg-[#38ef9b] hover:shadow-[0_0_20px_rgba(46,216,138,0.35)] transition-all"
      >
        <LogIn className="w-4 h-4" />
        Register Account
      </Link>
    </div>
  );
}
